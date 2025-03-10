import yaml
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama  # Update import
import json

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
    # Update to use gemma:2b as default if model_name is not specified
    model_name = agent_config['llm'].get('model_name', 'gemma:2b')
    llm = Ollama(model=model_name)
    return Agent(
        role=agent_config['role'],
        goal=agent_config['goal'],
        backstory=agent_config['backstory'],
        verbose=agent_config.get('verbose', True),
        llm=llm
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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
    agents_config = load_config('config/agents.yaml')  # Update path
    tasks_config = load_config('config/tasks.yaml')    # Update path
    
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
    topic = "quantum computing" if len(sys.argv) < 2 else sys.argv[1]
    run_crew(topic)