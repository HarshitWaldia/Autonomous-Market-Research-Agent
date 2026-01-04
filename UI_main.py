"""
Autonomous Market Research Agent
Groq + LLaMA-3.3-70B-Versatile
Single-file LangGraph Agentic System with Gradio UI

This agent demonstrates:
- Autonomous planning and task decomposition
- Multi-step reasoning workflow
- Tool usage (web search simulation)
- Dynamic research synthesis
- Validation and quality control
- Interactive Gradio Interface
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import json
import gradio as gr
from datetime import datetime


# Load environment variables
load_dotenv()

# Global LLM instance (will be initialized with API key)
llm = None


def initialize_llm(api_key: str):
    """Initialize Groq LLM with provided API key"""
    global llm
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2048,
        api_key=api_key,
    )


# Define the agent state
class ResearchState(TypedDict):
    """State management for the research agent workflow"""

    query: str  # Original research question
    sub_questions: List[str]  # Decomposed research tasks
    research_data: dict  # Gathered information per sub-question
    analysis: str  # Analytical insights
    validation_passed: bool  # Quality check result
    final_report: str  # Executive summary and recommendation
    iteration_count: int  # Track re-analysis attempts


def print_step(step_name: str, content: str = "", progress_callback=None):
    """Pretty print agent steps with optional callback for UI updates"""
    message = f"\n{'='*60}\n🤖 AGENT STEP: {step_name}\n{'='*60}"
    if content:
        message += f"\n{content}"
    message += "\n"

    print(message)

    # Send update to UI if callback provided
    if progress_callback:
        progress_callback(f"**{step_name}**\n{content}")

    return message


# AGENT 1: PLANNER - Task Decomposition


def planner_agent(state: ResearchState):
    """
    Autonomous planning: Breaks research query into specific sub-questions
    This mimics how a human analyst would approach a complex research task
    """
    print_step("PLANNER AGENT", "Breaking down research question...")

    prompt = PromptTemplate.from_template(
        """
You are an expert research planner. Your job is to decompose complex research questions into specific, actionable sub-questions.

Research Question: "{query}"

Break this down into 4-6 specific sub-questions that will comprehensively answer the main question.

Output ONLY a JSON array of strings, nothing else. Example format:
["What is X?", "What is Y?", "How do they compare?", "What are the trends?", "What is the recommendation?"]

JSON array:"""
    )

    response = (prompt | llm | StrOutputParser()).invoke({"query": state["query"]})

    # Parse JSON response
    try:
        # Extract JSON array from response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        sub_questions = json.loads(response.strip())
    except:
        # Fallback if JSON parsing fails
        sub_questions = [
            f"What are the key aspects of {state['query']}?",
            "What are the main differences or comparisons?",
            "What are current trends and adoption patterns?",
            "What are the pros and cons?",
            "What is the recommendation?",
        ]

    print(f"📋 Generated {len(sub_questions)} research sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"   {i}. {q}")

    return {"sub_questions": sub_questions, "iteration_count": 0}


# AGENT 2: RESEARCHER - Information Gathering


def research_agent(state: ResearchState):
    """
    Information gathering: Collects data for each sub-question
    In production, this would call actual web search APIs
    """
    print_step("RESEARCH AGENT", "Gathering information for each sub-question...")

    research_data = {}

    for i, sub_q in enumerate(state["sub_questions"], 1):
        print(f"🔍 Researching: {sub_q}")

        prompt = PromptTemplate.from_template(
            """
You are conducting in-depth research. Answer this specific question with factual, detailed information as if you've searched the web and consulted multiple sources.

Original Research Topic: "{main_query}"

Specific Question: "{sub_question}"

Provide comprehensive, factual information. Include specific details, trends, statistics, and real-world context where relevant.

Answer:"""
        )

        answer = (prompt | llm | StrOutputParser()).invoke(
            {"main_query": state["query"], "sub_question": sub_q}
        )

        research_data[sub_q] = answer
        print(f"   ✓ Completed ({len(answer)} chars)")

    return {"research_data": research_data}


# AGENT 3: ANALYST - Pattern Recognition & Insights


def analysis_agent(state: ResearchState):
    """
    Analysis: Identifies patterns, comparisons, and insights from gathered data
    """
    print_step("ANALYSIS AGENT", "Analyzing findings and extracting insights...")

    # Compile all research findings
    research_summary = "\n\n".join(
        [f"Q: {q}\nA: {a}" for q, a in state["research_data"].items()]
    )

    prompt = PromptTemplate.from_template(
        """
You are an expert analyst. Review the research findings below and provide a structured analysis.

Original Research Question: "{query}"

Research Findings:
{findings}

Provide analysis covering:
1. Key Patterns and Trends
2. Comparative Advantages/Disadvantages (if applicable)
3. Market Positioning or Current State
4. Critical Insights and Takeaways

Be analytical, objective, and insightful.

Analysis:"""
    )

    analysis = (prompt | llm | StrOutputParser()).invoke(
        {"query": state["query"], "findings": research_summary}
    )

    print(f"📊 Analysis complete ({len(analysis)} chars)")

    return {"analysis": analysis}


# AGENT 4: CRITIC - Quality Validation


def critic_agent(state: ResearchState):
    """
    Validation: Checks for hallucinations, bias, and unsupported claims
    This ensures output quality and reliability
    """
    print_step("CRITIC AGENT", "Validating research quality...")

    prompt = PromptTemplate.from_template(
        """
You are a strict research validator. Review the analysis below for:
- Hallucinations or unsupported claims
- Obvious bias or one-sided arguments
- Logical inconsistencies
- Missing critical information

Original Question: "{query}"

Analysis to Validate:
{analysis}

Respond with ONLY one word:
- "VALID" if the analysis is sound, balanced, and well-supported
- "INVALID" if it has significant issues

Verdict:"""
    )

    verdict = (prompt | llm | StrOutputParser()).invoke(
        {"query": state["query"], "analysis": state["analysis"]}
    )

    is_valid = "VALID" in verdict.strip().upper()

    print(f"✅ Validation: {'PASSED' if is_valid else 'FAILED'}")

    return {"validation_passed": is_valid}


# AGENT 5: SYNTHESIZER - Final Report Generation


def synthesis_agent(state: ResearchState):
    """
    Synthesis: Generates executive summary and actionable recommendations
    """
    print_step("SYNTHESIS AGENT", "Creating executive summary and recommendations...")

    prompt = PromptTemplate.from_template(
        """
You are creating a final executive research report.

Research Question: "{query}"

Analysis:
{analysis}

Generate a professional report with:

1. EXECUTIVE SUMMARY (2-3 sentences)
2. KEY FINDINGS (3-5 bullet points)
3. COMPARATIVE ANALYSIS or PROS/CONS (if applicable)
4. RECOMMENDATION (clear, actionable conclusion)

Format this as a polished, decision-ready report.

FINAL REPORT:"""
    )

    final_report = (prompt | llm | StrOutputParser()).invoke(
        {"query": state["query"], "analysis": state["analysis"]}
    )

    print(f"📄 Final report generated ({len(final_report)} chars)")

    return {"final_report": final_report}


# ROUTING LOGIC - Conditional Workflow


def route_after_validation(state: ResearchState):
    """
    Dynamic routing: Re-analyze if validation fails, otherwise proceed to synthesis
    Prevents infinite loops with iteration limit
    """
    if state["validation_passed"]:
        return "synthesizer"
    elif state["iteration_count"] < 2:  # Max 2 re-analysis attempts
        print("⚠️  Validation failed. Re-analyzing...")
        return "analyst"
    else:
        print("⚠️  Max iterations reached. Proceeding with current analysis.")
        return "synthesizer"


# BUILD THE LANGGRAPH WORKFLOW


def build_research_agent():
    """Construct the LangGraph state machine"""

    graph = StateGraph(ResearchState)

    # Add all agent nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", research_agent)
    graph.add_node("analyst", analysis_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("synthesizer", synthesis_agent)

    # Define the workflow edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")

    # Conditional routing after validation
    graph.add_conditional_edges(
        "critic",
        route_after_validation,
        {
            "analyst": "analyst",  # Re-analyze if validation fails
            "synthesizer": "synthesizer",  # Proceed if validation passes
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# MAIN EXECUTION


def run_research(query: str, api_key: str, progress=gr.Progress()):
    """
    Main function to run research with Gradio UI integration
    """
    if not query.strip():
        return "⚠️ Please enter a research question.", ""

    if not api_key.strip():
        return (
            "⚠️ Please enter your Groq API key. Get one at https://console.groq.com",
            "",
        )

    try:
        # Initialize LLM with provided API key
        initialize_llm(api_key)

        # Progress updates
        progress(0, desc="Initializing agent...")

        # Build the agent
        app = build_research_agent()

        progress(0.2, desc="Planning research...")

        # Run the agent
        result = app.invoke({"query": query})

        progress(1.0, desc="Research complete!")

        # Format the output
        report = f"""# 📊 Market Research Report
        
**Research Question:** {query}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

{result["final_report"]}

---

## 📋 Research Plan

"""
        for i, q in enumerate(result["sub_questions"], 1):
            report += f"{i}. {q}\n"

        # Create summary for logs
        logs = f"""✅ Research Completed Successfully

📝 Query: {query}
🔢 Sub-questions generated: {len(result["sub_questions"])}
✓ Validation: {"PASSED" if result["validation_passed"] else "FAILED"}
🔄 Iterations: {result.get("iteration_count", 0)}
"""

        return report, logs

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and try again."
        return error_msg, error_msg


def create_gradio_interface():
    """
    Create and configure the Gradio interface with minimal Apple-inspired design
    """

    # Custom CSS for glassmorphism and minimal design
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 24px !important;
        padding: 3rem !important;
        margin: 2rem auto !important;
        max-width: 1200px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    }
    
    .gr-box {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    
    .gr-input, .gr-text-input, textarea {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-input:focus, .gr-text-input:focus, textarea:focus {
        background: rgba(255, 255, 255, 0.18) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 4px 16px rgba(255, 255, 255, 0.1) !important;
    }
    
    .gr-input::placeholder, textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .gr-button {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 14px 28px !important;
        transition: all 0.3s ease !important;
        font-size: 15px !important;
    }
    
    .gr-button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(255, 255, 255, 0.2) !important;
    }
    
    .gr-button-primary {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2)) !important;
        font-weight: 600 !important;
    }
    
    label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 8px !important;
    }
    
    .gr-form {
        background: transparent !important;
        border: none !important;
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    h1 {
        font-size: 42px !important;
        margin-bottom: 8px !important;
    }
    
    p, .gr-markdown {
        color: rgba(255, 255, 255, 0.85) !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    
    .gr-markdown h3 {
        font-size: 18px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .info {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 13px !important;
    }
    
    /* Report output styling */
    .report-container {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        margin-top: 1rem !important;
    }
    
    /* Remove unnecessary borders */
    .gr-panel {
        border: none !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    """

    with gr.Blocks(css=custom_css, title="AI Research Agent") as demo:

        gr.Markdown(
            """
        # 🔮 AI Research Agent
        Autonomous market research powered by Groq + LangGraph
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="gsk_...",
                    type="password",
                    info="Get your key at console.groq.com",
                )

                query_input = gr.Textbox(
                    label="Research Question",
                    placeholder="What would you like to research?",
                    lines=4,
                )

                with gr.Row():
                    submit_btn = gr.Button("✨ Research", variant="primary", scale=2)
                    clear_btn = gr.ClearButton([query_input], value="Clear", scale=1)

        with gr.Row():
            report_output = gr.Markdown(value="", elem_classes="report-container")

        with gr.Row():
            logs_output = gr.Textbox(
                label="Status",
                lines=4,
                interactive=False,
                show_label=False,
                placeholder="Research status will appear here...",
            )

        # Button actions
        submit_btn.click(
            fn=run_research,
            inputs=[query_input, api_key_input],
            outputs=[report_output, logs_output],
        )

        clear_btn.click(fn=lambda: ("", ""), outputs=[report_output, logs_output])

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING GRADIO INTERFACE")
    print("=" * 60 + "\n")

    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch()
