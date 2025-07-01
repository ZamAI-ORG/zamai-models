#!/usr/bin/env python3
"""
Natural Language Understanding Engine for ZamAI
Specialized for Pashto language processing
"""
import os
import json

class NLUEngine:
    """
    Advanced NLU with cultural and contextual awareness for Pashto
    """
    
    def __init__(self, client, model_id="tasal9/ZamAI-Mistral-7B-Pashto"):
        """
        Initialize the NLU engine
        
        Args:
            client: HFInferenceClient object
            model_id: Model ID for language understanding
        """
        self.client = client
        self.model_id = model_id
        
        # Cultural context templates
        self.system_prompts = {
            "general": "تاسو د پښتو ژبې یو ګټور مرستیال یاست. د افغان کلتور په درناوي سره ځواب ورکړئ.",
            "educational": "تاسو د پښتو ژبې ښوونکی یاست. د زده کونکو سره صبر وکړئ او ښه تشریح ورکړئ.",
            "cultural": "د افغانستان د کلتور، تاریخ او دودونو په اړه مالومات ورکړئ. د اسلامي ارزښتونو درناوی وکړئ."
        }
        
        # Intent classification categories
        self.intent_categories = [
            "question", "request", "greeting", "farewell", 
            "thanks", "complaint", "opinion", "statement"
        ]

    def classify_intent(self, text):
        """
        Classify the intent of the user's text
        
        Args:
            text: User input text
        
        Returns:
            Classified intent and confidence
        """
        # Use zero-shot classification for intent detection
        result = self.client.zero_shot_classify(
            text=text,
            model="tasal9/Multilingual-ZamAI-Embeddings",
            candidate_labels=self.intent_categories
        )
        
        # Get the most likely intent
        if result and "labels" in result and "scores" in result:
            top_intent = result["labels"][0]
            confidence = result["scores"][0]
            return top_intent, confidence
        
        return "statement", 0.5

    def get_embedding(self, text):
        """
        Get vector embedding for the text
        
        Args:
            text: Input text
        
        Returns:
            Vector embedding
        """
        return self.client.embed_text(
            text=text,
            model="tasal9/Multilingual-ZamAI-Embeddings"
        )

    def generate(self, prompt, context_type="general", temperature=0.7, max_new_tokens=512, stop_sequences=None):
        """
        Generate a response based on the prompt
        
        Args:
            prompt: User input prompt
            context_type: Type of cultural context
            temperature: Creativity parameter
            max_new_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
        
        Returns:
            Generated response
        """
        # Add cultural context with system prompt
        system_prompt = self.system_prompts.get(context_type, self.system_prompts["general"])
        
        # Classify intent for better response targeting
        intent, _ = self.classify_intent(prompt.split("User: ")[-1].split("\nAssistant:")[0])
        
        # Prepare full prompt with context
        full_prompt = f"<s>[INST] {system_prompt}\n\nIntent: {intent}\n\n{prompt} [/INST]"
        
        try:
            # Call the HF client to generate text
            result = self.client.llm_prompt(
                prompt=full_prompt,
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=stop_sequences or []
            )
            
            # Clean up the response
            response = result.strip()
            
            # Remove any extra system or formatting text that might have leaked
            if "[/INST]" in response:
                response = response.split("[/INST]")[1]
                
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return "زه بښنه غواړم، زه نشم کولی ستاسو پوښتنې ته ځواب ورکړم."  # I'm sorry, I can't answer your question
