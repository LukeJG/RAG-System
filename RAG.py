import os
import re
import textwrap
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from transformers.utils import is_flash_attn_2_available

from huggingface_hub import notebook_login

notebook_login()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
    
    @staticmethod
    def text_formatter(text: str) -> str:
        cleaned_text = text.replace("\n", " ").strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            pages_and_texts = []
            
            for page_number, page in tqdm(enumerate(doc), desc="Processing pages"):
                text = page.get_text()
                text = self.text_formatter(text=text)
    
                if not text.strip():
                    continue
                
                page_info = {
                    "page_number": page_number,
                    "page_char_count": len(text),
                    "page_word_count": len(text.split()),
                    "page_sentence_count_raw": len(text.split(". ")),
                    "page_token_count": len(text) / 4,  # Approximate: 1 token ≈ 4 characters
                    "text": text
                }
                pages_and_texts.append(page_info)
            
            doc.close()
            logger.info(f"Successfully processed {len(pages_and_texts)} pages")
            return pages_and_texts
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def extract_sentences(self, pages_and_texts: List[Dict]) -> List[Dict]:
        logger.info("Extracting sentences using spaCy")
        
        for item in tqdm(pages_and_texts, desc="Extracting sentences"):
            sentences = list(self.nlp(item["text"]).sents)
            item["sentences"] = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
            item["page_sentence_count_spacy"] = len(item["sentences"])
        
        return pages_and_texts


class TextChunker:   
    def __init__(self, chunk_size: int = 10, min_token_length: int = 30):
        self.chunk_size = chunk_size
        self.min_token_length = min_token_length
    
    @staticmethod
    def split_list(input_list: List[str], slice_size: int) -> List[List[str]]:
        return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]
    
    def create_chunks(self, pages_and_texts: List[Dict]) -> List[Dict]:
        logger.info(f"Creating chunks with size {self.chunk_size}")
        
        pages_and_chunks = []
        
        for item in tqdm(pages_and_texts, desc="Creating chunks"):
            sentence_chunks = self.split_list(item["sentences"], self.chunk_size)
            
            for sentence_chunk in sentence_chunks:
                joined_chunk = " ".join(sentence_chunk).strip()
                joined_chunk = re.sub(r'\s+', ' ', joined_chunk)
                joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
                
                chunk_dict = {
                    "page_number": item["page_number"],
                    "sentence_chunk": joined_chunk,
                    "chunk_char_count": len(joined_chunk),
                    "chunk_word_count": len(joined_chunk.split()),
                    "chunk_token_count": len(joined_chunk) / 4
                }
                
                pages_and_chunks.append(chunk_dict)
        
        filtered_chunks = [
            chunk for chunk in pages_and_chunks 
            if chunk["chunk_token_count"] > self.min_token_length
        ]
        
        logger.info(f"Created {len(filtered_chunks)} chunks (filtered from {len(pages_and_chunks)})")
        return filtered_chunks


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Initialized embedding model '{model_name}' on device '{device}'")
    
    def generate_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> Tuple[List[Dict], torch.Tensor]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        texts = [chunk["sentence_chunk"] for chunk in chunks]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].cpu().numpy()
        
        return chunks, embeddings


class GPUMemoryManager:
    @staticmethod
    def get_gpu_memory_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0

        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        return round(gpu_memory_bytes / (2**30))

    @staticmethod
    def get_model_recommendation(gpu_memory_gb: float) -> Tuple[str, bool]:
        if gpu_memory_gb < 5.1:
            logger.warning(f"GPU memory ({gpu_memory_gb}GB) may be insufficient for local LLM")
            return "google/gemma-2b-it", True
        elif gpu_memory_gb < 8.1:
            logger.info(f"GPU memory: {gpu_memory_gb}GB | Recommended: Gemma 2B (4-bit)")
            return "google/gemma-2b-it", True
        elif gpu_memory_gb < 19.0:
            logger.info(f"GPU memory: {gpu_memory_gb}GB | Recommended: Gemma 2B (float16)")
            return "google/gemma-2b-it", False
        else:
            logger.info(f"GPU memory: {gpu_memory_gb}GB | Recommended: Gemma 7B")
            return "google/gemma-7b-it", False


class LLMManager:
    def __init__(self, model_id: Optional[str] = None, use_quantization: Optional[bool] = None):
        self.gpu_memory_gb = GPUMemoryManager.get_gpu_memory_gb()

        if model_id is None or use_quantization is None:
            self.model_id, self.use_quantization = GPUMemoryManager.get_model_recommendation(
                self.gpu_memory_gb
            )
        else:
            self.model_id = model_id
            self.use_quantization = use_quantization

        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"LLM Configuration: {self.model_id}, Quantization: {self.use_quantization}")

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.use_quantization:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    def _get_attention_implementation(self) -> str:
        if (is_flash_attn_2_available() and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8):
            return "flash_attention_2"
        return "sdpa"  # scaled dot product attention

    def load_model(self):
        try:
            logger.info(f"Loading model: {self.model_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = self._setup_quantization_config()
            attn_implementation = self._get_attention_implementation()

            logger.info(f"Using attention implementation: {attn_implementation}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                device_map="auto" if self.use_quantization else None
            )

            if not self.use_quantization and torch.cuda.is_available():
                self.model.to(self.device)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

 #   def generate_response(self, prompt: str, **generation_kwargs) -> str:
 #       if self.model is None or self.tokenizer is None:
 #           raise ValueError("Model not loaded. Call load_model() first.")

 #       default_params = {
 #           "max_new_tokens": 512,
 #           "temperature": 0.7,
 #           "do_sample": True,
 #           "top_p": 0.9,
 #           "top_k": 50,
 #           "repetition_penalty": 1.1,
 #           "pad_token_id": self.tokenizer.eos_token_id
 #       }
 #       default_params.update(generation_kwargs)

 #       inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
 #       inputs = {k: v.to(self.device) for k, v in inputs.items()}

 #       with torch.no_grad():
 #           outputs = self.model.generate(**inputs, **default_params)

 #       response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
 #       return response.replace(prompt, "").strip()


    def generate_response(self, prompt: str, **generation_kwargs) -> str:
      if self.model is None or self.tokenizer is None:
          raise ValueError("Model not loaded. Call load_model() first.")

      default_params = {
          "max_new_tokens": 512,
          "temperature": 0.7,
          "do_sample": True,
          "top_p": 0.9,
          "top_k": 50,
          "repetition_penalty": 1.1,
          "pad_token_id": self.tokenizer.eos_token_id,
          "eos_token_id": self.tokenizer.eos_token_id,
          "return_dict_in_generate": True,
          "output_scores": False
      }
      default_params.update(generation_kwargs)

      inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
      inputs = {k: v.to(self.device) for k, v in inputs.items()}

      input_length = inputs['input_ids'].shape[1]
 #     print(f"DEBUG: Input token length: {input_length}")
  #    print(f"DEBUG: Original prompt length (chars): {len(prompt)}")
   #   print(f"DEBUG: First 100 chars of prompt: {prompt[:100]}...")

      with torch.no_grad():
          outputs = self.model.generate(**inputs,
                                        output_attentions=False,
                                        output_hidden_states=False, 
                                        **default_params)


      generated_sequence = outputs.sequences[0] 

      output_length = generated_sequence.shape[0] 
    #  print(f"DEBUG: Output token length: {output_length}")
    #  print(f"DEBUG: New tokens generated: {output_length - input_length}")


      full_response = self.tokenizer.decode(generated_sequence.tolist(), skip_special_tokens=True) # Convert tensor to list
    #  print(f"DEBUG: Full response length (chars): {len(full_response)}")
    #  print(f"DEBUG: Full response first 200 chars: {full_response[:200]}...")

      if output_length > input_length:
          new_tokens = generated_sequence[input_length:] 
          new_response = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
          print(f"DEBUG: New tokens response: {new_response[:200]}...")

          replacement_response = full_response.replace(prompt, "").strip()
          print(f"DEBUG: String replacement response: {replacement_response[:200]}...")

          return new_response.strip()
      else:
          print("DEBUG: No new tokens generated!")
          return "No response generated"

class BankruptcyPromptManager:
    BANKRUPTCY_EXAMPLES = [
    #    {
   #         "query": "What assets are being sold in the case",
    #        "answer": "The assets being sold include property located at 109 W. Main St. (Suite C01), Durham, NC 27701 known as BALDWIN LOFTS CONDOS and (b) certain furniture, equipment, and other tangible personal property used in the Debtor’s restaurant operations."
     #   },
        {
            "query": "Was there a stalking horse bidder in the case?",
            "answer": "Yes, the court-approved bidding procedures designated XYZ Holdings, LLC as the stalking horse bidder with a purchase price of $1.25 million and a breakup fee of $50,000."
        },
        {
            "query": "How many qualified bidders participated in the auction?",
            "answer": "A total of four qualified bidders participated in the auction conducted under the court-approved bidding procedures."
        },
        {
            "query": "Did the Chapter 11 case convert to a Chapter 7 case?",
            "answer": "Yes, the bankruptcy case was converted from Chapter 11 to Chapter 7 on March 15, 2024, after the debtor failed to confirm a reorganization plan."
        },
        {
            "query": "Was the sale approved under Section 363 of the Bankruptcy Code?",
            "answer": "Yes, the sale was approved pursuant to Section 363(b) and 363(f) of the Bankruptcy Code, allowing the assets to be sold free and clear of liens, with liens transferring to the proceeds."
        },
        {
            "query": "Did the court approve any bid protections?",
            "answer": "Yes, the court approved bid protections including a breakup fee of $100,000 and an expense reimbursement of up to $25,000 for the stalking horse bidder."
        },
        {
            "query": "Was a liquidation analysis provided as part of the disclosure statement?",
            "answer": "Yes, the disclosure statement included a liquidation analysis comparing creditor recoveries under a Chapter 11 plan versus a Chapter 7 liquidation."
        },
   #     {
   #         "query": "Was the sale process competitive or was there only one bid?",
   #         "answer": "The sale process was competitive, with multiple qualified bids submitted. The final sale price exceeded the stalking horse bid by approximately 15%."
   #     },
  #      {
  #          "query": "Was creditor support disclosed for the proposed plan?",
  #          "answer": "Yes, the disclosure statement indicated that over 75% in amount and number of Class 3 general unsecured creditors supported the proposed Chapter 11 plan."
  #      },
 #       {
 #           "query": "Were any objections filed to the sale motion?",
 #           "answer": "Yes, objections were filed by the secured lender and the landlord, but the court overruled them following a hearing on February 20, 2024."
 #       },
#        {
#            "query": "Was the debtor operating post-petition or was a trustee appointed?",
#            "answer": "A Chapter 11 trustee was appointed on October 10, 2023, due to concerns about mismanagement and potential fraud raised by the U.S. Trustee."
#        }
    ]

    FINANCIAL_KEYWORDS = [
        "assets", "liabilities", "creditors", "debtors", "liquidation", "reorganization",
        "discharge", "automatic stay", "trustee", "secured debt", "unsecured debt",
        "preference payments", "fraudulent transfers", "exemptions", "means test",
        "disposable income", "bankruptcy estate", "adversary proceedings"
    ]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_bankruptcy_prompt(self, query: str, context_items: List[Dict]) -> str:
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        examples_text = ""
        for i, example in enumerate(self.BANKRUPTCY_EXAMPLES, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Query: {example['query']}\n"
            examples_text += f"Answer: {example['answer']}\n"

        base_prompt = f"""You are a bankruptcy analytics expert. Based on the following context items from court documents, sale orders, or case dockets, answer the query with clarity, precision, and a focus on factual analysis.
        Extract and synthesize key procedural and financial details directly from the context. Your goal is to identify what occurred in the case — such as asset sales, plan outcomes, bidder activity, and trustee actions — and summarize those developments in a way useful for analysts, financial professionals, or legal case reviewers.
        Use the following examples as reference for the ideal answer style:
        {examples_text}
        Guidelines for responses:
        - Be concise but thorough; only include facts supported by the context
        - Use neutral, objective tone without legal interpretation unless specified
        - Focus on activity and outcomes: sales, conversions, filings, approvals, objections, etc.
        - Include relevant figures (dates, amounts, bidder count, percentages) when available
        - Do not speculate — if data is unclear or missing, say so explicitly

        Context items: {context}

        User query: {query}

        Based on the above context, provide a clear, data-driven answer:"""

        dialogue_template = [{"role": "user", "content": base_prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation=dialogue_template,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt



class BankruptcyRAGSystem:
    """Complete RAG system specialized for bankruptcy and finance."""

    # Sample bankruptcy-focused queries
    SAMPLE_QUERIES = [
        "How many qualified bidders participated in the asset auction?",
        "Was a stalking horse bidder designated in this case?",
        "Did the bankruptcy case convert from Chapter 11 to Chapter 7?",
        "Was the sale approved under Section 363 of the Bankruptcy Code?",
        "What was the final purchase price for the debtor's assets?",
        "Were there any objections filed to the sale or plan confirmation?",
        "What bid protections were granted to the stalking horse bidder?",
        "Did the debtor operate as a debtor-in-possession or was a trustee appointed?",
        "What classes of creditors voted in favor of the plan?",
        "Was the disclosure statement approved by the court?",
        "Was a liquidation analysis included in the disclosure statement?",
        "What were the estimated recoveries for unsecured creditors under the plan?",
        "Did the plan provide for substantive consolidation of debtor entities?",
        "Were insider transactions disclosed or challenged during the proceedings?",
        "Was DIP (Debtor-in-Possession) financing approved by the court?",
        "Did any creditors file motions to lift the automatic stay?",
        "Was a Section 506 valuation performed on secured claims?",
        "Did the debtor file monthly operating reports (MORs) consistently?",
        "What were the projected vs. actual distributions under the confirmed plan?",
        "Was there a motion to appoint an examiner or convert the case?"
    ]


    def __init__(self, embedding_model_name: str = "all-mpnet-base-v2",
                 llm_model_id: Optional[str] = None,
                 chunk_size: int = 10,
                 min_token_length: int = 30):

        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(chunk_size, min_token_length)
        self.embedding_generator = EmbeddingGenerator(embedding_model_name)

        self.llm_manager = LLMManager(llm_model_id)
        self.prompt_manager = None

        self.chunks = None
        self.embeddings = None

        logger.info("BankruptcyRAGSystem initialized")

    def setup_system(self):
        logger.info("Setting up LLM components...")
        self.llm_manager.load_model()
        self.prompt_manager = BankruptcyPromptManager(self.llm_manager.tokenizer)
        logger.info("System setup complete")

    def process_document(self, pdf_path: str):
        logger.info(f"Processing bankruptcy document: {pdf_path}")

        pages_and_texts = self.pdf_processor.extract_text_from_pdf(pdf_path)
        pages_and_texts = self.pdf_processor.extract_sentences(pages_and_texts)

        self.chunks = self.text_chunker.create_chunks(pages_and_texts)

        self.chunks, self.embeddings = self.embedding_generator.generate_embeddings(self.chunks)

        logger.info("Document processing complete")

    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context for a query."""
        if self.chunks is None or self.embeddings is None:
            raise ValueError("No document processed. Call process_document() first.")

        query_embedding = self.embedding_generator.model.encode(
            query, convert_to_tensor=True
        )

        similarities = util.dot_score(query_embedding, self.embeddings)[0]

        top_results = torch.topk(similarities, k=min(top_k, len(self.chunks)))

        context_items = []
        for score, idx in zip(top_results.values, top_results.indices):
            item = self.chunks[idx].copy()
            item["relevance_score"] = float(score)
            context_items.append(item)

        return context_items

    def answer_query(self, query: str,
                    top_k: int = 5,
                    use_bankruptcy_prompt: bool = True,
                    **generation_kwargs) -> Dict:

        if self.llm_manager.model is None:
            raise ValueError("LLM not loaded. Call setup_system() first.")

        context_items = self.retrieve_relevant_context(query, top_k)

        if use_bankruptcy_prompt:
            prompt = self.prompt_manager.create_bankruptcy_prompt(query, context_items)
        #else:
        #    prompt = self.prompt_manager.create_general_financial_prompt(query, context_items)

        answer = self.llm_manager.generate_response(prompt, **generation_kwargs)

        return {
            "query": query,
            "answer": answer,
            "context_items": context_items,
            "num_context_items": len(context_items),
            "prompt_type": "bankruptcy" if use_bankruptcy_prompt else "general"
        }


    def batch_evaluate(self, queries: Optional[List[str]] = None, num_queries: int = 5) -> List[Dict]:
        if queries is None:
            queries = random.sample(self.SAMPLE_QUERIES, min(num_queries, len(self.SAMPLE_QUERIES)))

        results = []
        for query in tqdm(queries, desc="Processing queries"):
            try:
                result = self.answer_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                continue

        return results

    def get_system_stats(self) -> Dict:
        stats = {
            "gpu_memory_gb": self.llm_manager.gpu_memory_gb,
            "model_id": self.llm_manager.model_id,
            "uses_quantization": self.llm_manager.use_quantization,
            "device": self.llm_manager.device,
        }

        if self.chunks:
            chunk_df = pd.DataFrame(self.chunks)
            stats.update({
                "total_chunks": len(self.chunks),
                "avg_tokens_per_chunk": chunk_df["chunk_token_count"].mean(),
                "min_tokens": chunk_df["chunk_token_count"].min(),
                "max_tokens": chunk_df["chunk_token_count"].max(),
            })

        return stats
    

def main():
    rag_system = BankruptcyRAGSystem()
    rag_system.setup_system()

    rag_system.process_document("")

    result = rag_system.answer_query("Give me a list of the assets that will be sold in the case")
    print(result['answer'])

