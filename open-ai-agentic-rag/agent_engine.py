import json
from typing import Dict, Any
from core_llm import CustomAgentScratchLLM
from tools import AgentTools
from rag_engine import LocalRAGEngine

class CognitiveAgentEngine:
    """Autonomous loop running Parallel Fusion over LLM attention blocks and RAG matrices."""
    def __init__(self):
        self.model_registry = {
            "scratch-agent-v1": CustomAgentScratchLLM("scratch-agent-v1"),
            "scratch-agent-smarter-v2": CustomAgentScratchLLM("scratch-agent-smarter-v2")
        }

    def process_request(self, query: str, model_id: str) -> Dict[str, Any]:
        # 1. Normalize the model input parameter keys safely
        normalized_id = str(model_id).lower().strip().replace("_", "-")
        target_key = "scratch-agent-smarter-v2" if ("v2" in normalized_id or "smarter" in normalized_id) else "scratch-agent-v1"
        selected_model = self.model_registry[target_key]
        
        # 2. RUN RAG IN PARALLEL: Scan knowledge database vector spaces first
        rag_context, rag_score = LocalRAGEngine.retrieve_context(query, "knowledge_base.txt")
        has_rag_context = (rag_score > 0.0)

        # 3. CONTEXT AUGMENTATION STEP: Formulate unified attention token string frame
        if has_rag_context:
            augmented_prompt = f"Context: {rag_context} Question: {query}"
            print(f"🧠 [Parallel Fusion] Augmented input context frame with local reference documentation.")
        else:
            augmented_prompt = query

        # 4. LLM COGNITIVE PASS: Execute matrix self-attention loops over fully augmented string
        logits = selected_model.analyze_intent(augmented_prompt)
        target_tool = max(logits, key=logits.get)
        llm_confidence = logits[target_tool]
        has_tool_intent = (llm_confidence > 0.0)

        # 5. ROUTING DECISION MATRIX DISPATCHER
        
        # Scenario A: Total System Miss (Both modules return zero validation metrics)
        if not has_tool_intent and not has_rag_context:
            thought_process = "Query skipped all tool parameters and text vector alignment fell below parameters."
            structured_payload = {
                "thought": thought_process,
                "action": {
                    "call_function": "None",
                    "status": "failed"
                },
                "observation": {
                    "retrieved_context_chunk": "No additional context can be found for your query."
                },
                "final_answer": "I do not know the answer, Google is your friend!"
            }
            return {"final_answer": json.dumps(structured_payload, indent=2, ensure_ascii=False)}

        # Scenario B: Pure RAG Context Only (No tool triggers, but documentation matches)
        elif not has_tool_intent and has_rag_context:
            thought_process = f"Augmented prompt tracking analyzed. No tool triggered, processing solely via RAG context strings."
            structured_payload = {
                "thought": thought_process,
                "action": {
                    "call_function": "local_knowledge_base_rag_retrieval",
                    "status": "success"
                },
                "observation": {
                    "retrieved_context_chunk": rag_context
                },
                "final_answer": "Synthesized response from local internal document vectors."
            }
            return {"final_answer": json.dumps(structured_payload, indent=2, ensure_ascii=False)}

        # Scenario C: Fusion Path (Model triggers positive confidence AND RAG yields context data assets)
        else:
            # Check if a tool was executed or if it's a fallback route
            if has_tool_intent:
                print(f"🤖 [Agent Core] Thought: Tool match triggered on augmented context context layer.")
                print(f"🎯 [Agent Core] Activating registered tool wrapper: '{target_tool}'")
                tool_result = AgentTools.execute_tool_by_name(target_tool)
                action_name = target_tool
                final_answer_text = "Successfully fetched telemetry and fused response parameters with local documentation assets."
            else:
                tool_result = "N/A"
                action_name = "local_knowledge_base_rag_retrieval"
                final_answer_text = "Synthesized answer from augmented vector space tracking context metrics."

            thought_process = (
                f"Executed parallel analysis via model '{target_key}'. Prompt frame was augmented with "
                f"RAG asset chunks (Score: {rag_score:.4f}). Attention layer triggered execution path "
                f"for tool pattern token signature: '{target_tool}' (Softmax Confidence: {llm_confidence:.4f})."
            )

            structured_payload = {
                "thought": thought_process,
                "action": {
                    "call_function": action_name,
                    "status": "success"
                },
                "observation": {
                    "raw_data_recovered": tool_result,
                    "fused_background_context": rag_context if has_rag_context else "No structural background document match found."
                },
                "final_answer": final_answer_text
            }
            
            return {"final_answer": json.dumps(structured_payload, indent=2, ensure_ascii=False)}
