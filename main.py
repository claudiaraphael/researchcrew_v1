import yaml
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama  # Import Ollama from langchain_community
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config(file_path):
    """
    Load and parse a YAML configuration file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_agent(agent_config):
    """
    Create an agent based on the provided configuration.
    """
    # Configure Ollama with gemma:2b model by default
    llm = Ollama(
        model="gemma:2b",
        temperature=agent_config['llm'].get('temperature', 0.7),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    return Agent(
        role=agent_config['role'],
        goal=agent_config['goal'],
        backstory=agent_config['backstory'],
        verbose=agent_config.get('verbose', True),
        llm=llm,
        allow_delegation=agent_config.get('allow_delegation', True)
    )

def create_task(task_config, agents_dict):
    """
    Create a task based on the provided configuration and agents dictionary.
    """
    agent = agents_dict[task_config['agent']]
    return Task(
        description=task_config['description'],
        expected_output=task_config.get('expected_output', ''),
        agent=agent
    )

def create_crew(topic):
    """
    Create the crew by loading configurations, creating agents, and setting up tasks.
    """
    # Load configuration files
    agents_config = load_config('config/agents.yaml')
    tasks_config = load_config('config/tasks.yaml')
    
    # Create agents
    agents_dict = {}
    for agent_config in agents_config['agents']:
        agents_dict[agent_config['id']] = create_agent(agent_config)
    
    # Create tasks
    tasks = []
    for task_config in tasks_config['tasks']:
        tasks.append(create_task(task_config, agents_dict))
    
    # Create crew
    research_crew = Crew(
        agents=list(agents_dict.values()),
        tasks=tasks,
        verbose=True
    )
    
    return research_crew

def run_crew(topic):
    """
    Run the crew with the given topic.
    """
    crew = create_crew(topic)
    result = crew.kickoff(inputs={"topic": topic})
    print(result)

if __name__ == "__main__":
    import sys
    topic = "agentic workflow design" if len(sys.argv) < 2 else sys.argv[1]
    run_crew(topic)