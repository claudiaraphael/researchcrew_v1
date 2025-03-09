# Description: This file contains the code to create a crew of agents and tasks based on a configuration file.
import yaml
import os
from crewai import Agent, Task, Crew
from langchain.llms import Ollama

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Chose model
def create_agent(agent_config):
    # Set up LLM based on config
    llm = Ollama(model=agent_config['llm']['model_name'])
    
    # Create and return agent
    return Agent(
        role=agent_config['role'],
        goal=agent_config['goal'],
        backstory=agent_config['backstory'],
        verbose=agent_config.get('verbose', True),
        llm=llm
    )

def create_task(task_config, agents_dict):
    # Get the agent for this task
    agent = agents_dict[task_config['agent']]
    
    # Create and return task
    return Task(
        description=task_config['description'],
        expected_output=task_config.get('expected_output', ''),
        agent=agent
    )

def create_crew(topic):
    # Load configuration
    config = load_config('agents.yaml')
    tasks_config = load_config('Tasks.yaml')
    
    # Create agents
    agents_dict = {}
    for agent_config in config['agents']:
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
    crew = create_crew(topic)
    result = crew.kickoff(inputs={"topic": topic})
    print(result)

if __name__ == "__main__":
    import sys
    topic = "quantum computing" if len(sys.argv) < 2 else sys.argv[1]
    run_crew(topic)