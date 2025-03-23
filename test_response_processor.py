import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Mock Workers
class RAGWorker:
    async def execute(self, state):
        print(f"📚 RAG Processing:")
        print(f"  - Finding relevant debate topics for: {state['user_response']}")
        # Simulating RAG finding a relevant topic based on user's interest
        if "climate" in state['user_response'].lower():
            topic = "climate_change"
            context = "Debate about climate change impacts and solutions"
        elif "education" in state['user_response'].lower():
            topic = "education_reform"
            context = "Debate about modern education system reforms"
        else:
            topic = "technology_ethics"
            context = "Debate about ethical implications of emerging technologies"
            
        print(f"  - Selected topic: {topic}")
        print(f"  - Context: {context}")
        
        return {
            "topic": topic,
            "context": context,
            "relevant_sources": [
                "Recent academic papers",
                "Expert opinions",
                "Case studies"
            ],
            "suggested_points": [
                "Personal experience",
                "Observable impacts",
                "Future implications"
            ]
        }

class ArgumentClassifier:
    async def execute(self, state):
        print(f"🔍 Classifying personal story/argument:")
        print(f"  - Topic context: {state.get('topic')}")
        print(f"  - Story: {state['user_response']}")
        print(f"  - Checking against suggested points: {state.get('suggested_points')}")
        
        # Analyzing the personal story
        return {
            "argument_type": "personal_experience",
            "argument_strength": 0.85,
            "matches_suggested_points": True
        }

class EffectivenessEvaluator:
    async def execute(self, state):
        print(f"⭐ Evaluating story effectiveness...")
        print(f"  - Topic: {state.get('topic')}")
        print(f"  - Story type: {state.get('argument_type')}")
        print(f"  - Context: {state.get('context')}")
        
        # Evaluating personal story effectiveness
        return {
            "effectiveness_score": 0.82,
            "persuasion_technique": "personal_narrative",
            "topic_relevance": 0.9
        }

class DatabaseWorker:
    async def execute(self, state):
        print(f"💾 Storing in database:")
        print(f"  - Topic: {state.get('topic')}")
        print(f"  - Story type: {state.get('argument_type')}")
        print(f"  - Effectiveness: {state.get('effectiveness_score')}")
        print(f"  - Technique: {state.get('persuasion_technique')}")
        print(f"  - Relevance: {state.get('topic_relevance')}")
        return {"status": "stored"}

@dataclass
class ProcessingState:
    user_response: str
    # RAG Worker fields
    topic: Optional[str] = None
    context: Optional[str] = None
    relevant_sources: Optional[List[str]] = None
    suggested_points: Optional[List[str]] = None
    
    # Argument Classifier fields
    argument_type: Optional[str] = None
    argument_strength: Optional[float] = None
    matches_suggested_points: Optional[bool] = None
    
    # Effectiveness Evaluator fields
    effectiveness_score: Optional[float] = None
    persuasion_technique: Optional[str] = None
    topic_relevance: Optional[float] = None
    
    # Database fields
    status: Optional[str] = None

class ResponseProcessor:
    def __init__(self, main_worker):
        self.main_worker = main_worker
        self.pre_steps = []
        self.post_step = None
        # Store topic context
        self.current_topic = None
        self.current_context = None
        self.current_suggested_points = None
    
    def add_pre_step(self, worker):
        self.pre_steps.append(worker)
        return self
        
    def add_post_step(self, worker):
        self.post_step = worker
        return self

    async def process_topic_selection(self, topic_interest):
        """Phase 1: Topic Selection using RAG"""
        print("\n📋 Phase 1: Topic Selection")
        state = ProcessingState(user_response=topic_interest)
        
        # Only use RAG worker for topic selection
        rag_worker = self.pre_steps[0]  # RAG should be first pre-step
        try:
            result = await rag_worker.execute(state.__dict__)
            # Store topic context for future use
            self.current_topic = result["topic"]
            self.current_context = result["context"]
            self.current_suggested_points = result["suggested_points"]
            return result
        except Exception as e:
            print(f"Error in topic selection: {str(e)}")
            raise

    async def process_story(self, personal_story):
        """Phase 2: Process Personal Story"""
        if not self.current_topic:
            raise ValueError("No topic selected. Must run topic selection first.")
            
        print("\n📝 Phase 2: Story Processing")
        # Initialize state with stored topic context
        state = ProcessingState(
            user_response=personal_story,
            topic=self.current_topic,
            context=self.current_context,
            suggested_points=self.current_suggested_points
        )
        
        # Process with remaining pre-steps (skip RAG)
        for i, pre_step in enumerate(self.pre_steps[1:], 1):
            print(f"\nRunning pre-step {i}...")
            try:
                pre_result = await pre_step.execute(state.__dict__)
                state = ProcessingState(**{**state.__dict__, **pre_result})
            except Exception as e:
                print(f"Error in pre-step {i}: {str(e)}")
                raise
        
        # Run main worker
        print("\nRunning main worker...")
        try:
            result = await self.main_worker.execute(state.__dict__)
            state = ProcessingState(**{**state.__dict__, **result})
        except Exception as e:
            print(f"Error in main worker: {str(e)}")
            raise
        
        # Run post-step if configured
        if self.post_step:
            print("\nRunning post-step...")
            try:
                post_state = state.__dict__.copy()
                post_result = await self.post_step.execute(post_state)
                state = ProcessingState(**{**state.__dict__, **post_result})
            except Exception as e:
                print(f"Error in post-step: {str(e)}")
                raise
        
        print("\n✅ Processing complete!")
        return result

async def test_processor():
    # Set up processor
    processor = (
        ResponseProcessor(EffectivenessEvaluator())
            .add_pre_step(RAGWorker())
            .add_pre_step(ArgumentClassifier())
            .add_post_step(DatabaseWorker())
    )
    
    # Test cases: First message selects topic, second is the personal story
    test_cases = [
        {
            "topic_interest": "I'm interested in discussing climate change impacts",
            "personal_story": "Last summer, my neighborhood experienced unprecedented flooding. Our community came together to help affected families, and it really opened my eyes to the immediate impacts of changing weather patterns."
        },
        {
            "topic_interest": "I want to talk about education reform",
            "personal_story": "As a first-generation college student, I struggled with the traditional lecture format. But when one professor used project-based learning, it completely changed my understanding and engagement with the material."
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print("\n" + "="*50)
        print(f"Test Case {i}")
        print("="*50)
        
        # Phase 1: Topic Selection
        topic_result = await processor.process_topic_selection(case["topic_interest"])
        print(f"\nSelected Topic: {topic_result['topic']}")
        print(f"Context: {topic_result['context']}")
        
        # Phase 2: Process Personal Story
        try:
            story_result = await processor.process_story(case["personal_story"])
            print("\nStory Processing Result:")
            print(f"- Effectiveness Score: {story_result['effectiveness_score']}")
            print(f"- Persuasion Technique: {story_result['persuasion_technique']}")
            print(f"- Topic Relevance: {story_result['topic_relevance']}")
        except Exception as e:
            print(f"\nError processing story: {str(e)}")
        
        print("-"*50)

# Run the test
if __name__ == "__main__":
    asyncio.run(test_processor()) 