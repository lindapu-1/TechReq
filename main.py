from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from tools import save_table, save_table_as_image, save_domain_needs, extract_and_save_scores
from openai import OpenAI
from autogen_agentchat.ui import Console
from autogen_core.models import UserMessage
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
import json
import asyncio
import time
from dotenv import load_dotenv
import os
from datetime import datetime
from fastapi.testclient import TestClient
from elicitron.backend.backend import app
from typing import List, Dict, Union, Optional
from em import MetricsVisualizer
from contextlib import redirect_stdout
import io
import uuid
import argparse

#product = "Mini Travel Hair Dryers"
#hazard = "The handheld hair dryers lack an immersion protection device and can cause death or serious injury due to electrocution or shock if the hair dryers fall into water when plugged in. The hair dryers are in violation of the federal regulations and present a substantial product hazard."
#product = "Desktop Heaters"
#hazard = "The heater's fan can fail to turn on and cause the unit to overheat and ignite, posing fire and burn hazards."
#product = "Suspension Fork for a Mountain Bike"
#hazard = "The suspension fork can break during use, posing a risk of injury to the user."
#product = "Childrens multi-purpose bike helmets"
#hazard = "The helmets do not comply with the positional stability, dynamic strength of retention system, impact attenuation, and certification requirements in violation of the CPSC federal safety regulation for bicycle helmets. The helmets can fail to protect in the event of a crash, posing a risk of head injury."

# Product configuration
# ================================User Input================================
product = "Folding Knives"
hazard = "The knives can be dangerous when used improperly, posing a risk of injury to the user."
model_steering = f"There is a reported hazard in the product which is {hazard}"
number_of_stakeholders_generated = 5
number_of_stakeholders_to_use = 3
# ================================End of User Input================================

# Initialize paths and timestamps
os.makedirs('multi-agent/domain_needs', exist_ok=True)

# Initialize clients
load_dotenv()
elicitron_client = TestClient(app)
visualizer = MetricsVisualizer()
f = io.StringIO()

async def run_elicitron(result_folder):
    stakeholders = []
    interviews = []
    questions = [
        f"What challenges do you face with current {product}?",
        f"How would an {product} best benefit you?",
        f"What features would you expect in such a {product}?",
    ]
    print('-----------creating stakeholders-----------')
    # Create stakeholders
    payload = {
        "design_context": product,
        "model_steering": model_steering,
        "number_of_stakeholders": number_of_stakeholders_generated
    }

    response = elicitron_client.post("/create_stakeholder_serial", json=payload)
    data = response.json()
    print("Created Stakeholders (Serial):", data)
    stakeholders = data["stakeholders"]


    # Filter stakeholders if needed
    if len(stakeholders) > number_of_stakeholders_to_use:
        print('-----------filtering stakeholders-----------')
        for s in stakeholders:
            if 'id' not in s:
                s['id'] = str(uuid.uuid4())
        
        payload = {
            "stakeholders": stakeholders,
            "n": number_of_stakeholders_to_use
        }
        response = elicitron_client.post("/filter_top_diverse", json=payload)
        data = response.json()
        stakeholders = data

    # Conduct interviews
    print('-----------conducting interviews-----------')
    for stakeholder in stakeholders:
        if 'id' not in stakeholder:
            stakeholder['id'] = str(uuid.uuid4())
        
        payload = {
            "stakeholder": stakeholder,
            "design_context": product,
            "questions": questions
        }
        response = elicitron_client.post("/conduct_interview", json=payload)
        interview_data = response.json()
        interviews.append(interview_data)

    # Generate report
    print('-----------generating report-----------')
    payload = {
        "interviews": interviews,
        "questions": questions
    }
    response = elicitron_client.post("/create_report", json=payload)
    final_report = response.json()

    # Extract customer needs
    print('-----------extracting customer needs-----------')
    payload = {
        "product": product,
        "report": final_report
    }
    response = elicitron_client.post("/extract_customer_needs", json=payload)
    needs_data = response.json()
    
    # Save all elicitron results
    print('-----------saving elicitron results-----------')
    elicitron_results = {
        "product": product,
        "model_steering": model_steering,
        "stakeholders": stakeholders,
        "interviews": interviews,
        "report": final_report,
        "questions": questions,
        "customer_needs": needs_data["customer_needs"]
    }
    
    # Save to result folder
    elicitron_output_path = os.path.join(result_folder, "elicitron_results.json")
    with open(elicitron_output_path, 'w') as f:
        json.dump(elicitron_results, f, indent=2)
    print('-----------elicitron part completed-----------')
    
    return needs_data

class InitialProductLeadAgent(AssistantAgent):
    def __init__(self, model_client, result_file_path, tools=[save_domain_needs]):
        super().__init__(
            name="initial_product_lead",
            model_client=model_client,
            tools=tools,
            system_message=f"""
            You are a helpful assistant that categorizes customer needs into specific engineering and management domains. 
            Given a list of needs, please categorize each need into the relevant domains: Mechanical Engineering, Material Science Engineering, Standards Manager, Industrial Engineering, Program Manager.
            Some needs may belong to more than one domain, so provide all applicable categories.

            Be as DOMAIN SPECIFIC as possible. For example, the material science engineer should only have materials related needs.
            Separate these needs out into domain categories for mechanical engineering, material science engineering, standards manager, industrial engineering, and program manager

            **IMPORTANT OUTPUT FORMAT:**  
                You **MUST** output a JSON object
                **IMPORTANT: DO NOT** add escape sequences (`\\n`, `\\"`, `\n`), or unnecessary formatting.  
                The output **must** be a raw JSON.

                **VALID EXAMPLE (Correct Format):**
                ```json
                "{{
                    "roles": [
                        {{
                            "role": "mechanical_engineer",
                            "needs": [
                                "need 1",
                                "need 2",
                                ...
                            ]
                        }},
                        {{
                            "role": "material_science_engineer",
                            "needs": [
                                "need 1",
                                "need 2",
                                ..
                            ]
                        }},
                        ...
                    ]
                }}"
                ```

            IMPORTANT:
            Pass EXACTLY this JSON **string** as the first argument, {result_file_path} as the second argument to the `save_domain_needs` tool.
            """
        )

class SupervisorAgent(AssistantAgent):
    def __init__(self, name, model_client, domain, json_filename, tools=[save_table]):
        super().__init__(
            name=name, 
            model_client=model_client, 
            tools=tools,
            system_message=f"""
            You are an supervior to a {domain} agent who is designing the {product}.
            Your task is to prompt the {domain} agent to generate a better table, and give feedback on the table, especially when you think the table is not good enough. You should examine each row of the table carefully.
            When you believe a good outcome has been arrived at between you and the intern, you can use the "SUBMIT" command to end the dialogue and submit the table. Note that you should first submit the final version of the table, and then use the SUBMIT command. Before you submit the table, you should double check whether each customer need is well translated into at least one metric.
            
            What is a good table:
            The first column is the customer need, the second column is the metric corresponding to this need, and the third column is the unit of this metric. If one customer need translates into multiple metrics, list them in multiple rows.
            Make sure the metrics are specific to the domain: {domain}.
            Make them as relevant to the product as possible.
            Do not be redundant.
            DO NOT include specific values in the metrics.

            **IMPORTANT OUTPUT FORMAT when submitting the table:**  
            You **MUST** output a JSON object as a **string**, not a Python dictionary.  
            **DO NOT** add escape sequences (e.g., `\\n`, `\\"`), or unnecessary formatting.  
            The output **must** be a raw JSON string.
            DO NOT put the whole output on one line.

            **VALID EXAMPLE (Correct Format):**
            ```json
            "{{
                \\"Need1\\": [
                    [\\"Metric1\\", \\"Unit1\\"],
                    [\\"Metric2\\", \\"Unit 2\\"]
                ],
                \\"Need 2\\": [
                    [\\"Metric 1\\", \\"Unit 1\\"],
                    [\\"Metric 2\\", \\"Unit 2\\"]
                ]
            }}"
            ```

            Pass EXACTLY {json_filename} as the first argument, and this JSON **string** as the second argument to the `save_table` tool.
            Only call the save_table tool when you proceed the submittion process. Respond with 'SUBMIT' once the files are successfully saved locally. The the conversation will terminate by your key word SUBMIT.
            """
            ) 

class MetricsAgent(AssistantAgent):
    def __init__(self, name, model_client, domain):
        super().__init__(
            name=name, 
            model_client=model_client, 
            system_message=f"""
            You are an expert {domain} who is designing the {product}.
            You are working with your supervisor, who will help you come up with an output, and you interact with them through dialogue.
            Your goal is to produce a metrics table as a part of the product design process that aligns with customer needs. You should aim for coming up with ideas and suggesting your plan to your supervisor. You should integrate the provided information and the feedback from the supervisor to keep polishing and improving the output. 
            Your task is to generate {domain}ing-specific metrics (WITH units) for given customer needs.
            The first column is the customer need, the second column is the metric corresponding to this need, and the third column is the unit of this metric. If one customer need translates into multiple metrics, list them in multiple rows.

            However, there are many challenges to come up with a list of metrics that are professional and relevant to the customer needs. So this is why you need to brainstorm with your product lead. You can suggest various metrics for each need and discuss whether some of them are good or not solid enough. One customer need might be translated into one or multiple metrics.

            You should dialogue with your product lead, for each customer need, you should examine the metrics and discuss whether they are good or not solid enough. When you propose a table, you should add an additional column to the table to indicate whether the metric is good, bad(need to improve or discard), or not solid enough. You should display your reason so that the product lead can make a decision.
            For each customer need, you should check whether need to 1) add more metrics, 2) remove some metrics, 3) improve some metrics.

            You can talk to your leader about which parts need to imporve and also ask for his feedback and comments. The leader is more experienced so may provide insights on how to improve. But you should always propose your own idea first.

            Make sure the metrics are specific to the domain: {domain}.
            Make them as relevant to the product as possible.
            Do not be redundant.
            DO NOT include specific values in the metrics.

            **IMPORTANT OUTPUT FORMAT:**  
            **DO NOT** add escape sequences (e.g., `\\n`, `\\"`), or unnecessary formatting.  
            DO NOT put the whole output on one line.
        
            """
            ) 
    
class ProductLeadAgent(AssistantAgent):
    def __init__(self, name, model_client, json_filename, tools=[save_table]):
        super().__init__(
            name=name, 
            model_client=model_client, 
            tools=tools,
            system_message=f"""
            You are an an product lead agent who is supervising the team of making {product}.
            Your team consist of agents in different domains. They already generated metrics for their domains. You role is to combine all the metrics into a final table.
            Their metrics may have overlaped or reduncancy, you need to identify them, make some deletion and changes, then combine them into a final table.
            Sincce you are the lead, you need to make sure the final table is comprehensive and not redundant, so that this can be passed to the engineer team.

            CRITICAL REQUIREMENT:
            - You MUST preserve the EXACT wording of ALL customer needs exactly as they appear in the input
            - DO NOT modify, simplify, or paraphrase any customer need
            - DO NOT combine or split customer needs


            **IMPORTANT OUTPUT FORMAT:**  
            You **MUST** output a JSON object as a **string**, not a Python dictionary.  
            **DO NOT** add escape sequences (`\\n`, `\\"`), or unnecessary formatting.  
            The output **must** be a raw JSON string.
            DO NOT put the whole output on one line.
            DO NOT change any wording of the customer needs.

            **VALID EXAMPLE (Correct Format):**
            ```json
            "{{
                \\"Need1\\": [
                    [\\"Metric1\\", \\"Unit1\\"],
                    [\\"Metric2\\", \\"Unit 2\\"]
                ],
                \\"Need 2\\": [
                    [\\"Metric 1\\", \\"Unit 1\\"],
                    [\\"Metric 2\\", \\"Unit 2\\"]
                ]
            }}"
            ```

            Pass EXACTLY {json_filename} as the first argument, and this JSON **string** as the second argument to the `save_table` tool.
            Only call the save_table tool when you proceed the submission process. Respond with 'SUBMIT' once the files are successfully saved locally. The the conversation will terminate by your key word SUBMIT.
            """
            ) 

class LLM_as_Judge(AssistantAgent):
    def __init__(self, name, model_client):
        super().__init__(
            name=name, 
            model_client=model_client, 
            system_message=f"""
                You are a judge evaluating the quality of engineering metrics for {product}. 
                Your role is to ensure the metrics are optimized for engineering implementation.

                Rate the table on a scale from 1 to 5 based on:

                1. Specificity (0-5 points):
                - Are metrics measurable with standard testing equipment?
                - Are units clearly defined and industry-standard?
                - Would different engineers get consistent results measuring these metrics?

                2. Relevance (0-5 points):
                - Does each category directly relate to product performance or user needs?
                - Are there any missing critical metrics?
                - Are there metrics that are too marketing-focused rather than engineering-focused?

                3. Non-Redundancy (0-5 points):
                - Are there metrics that measure similar properties?
                - Could any categories be merged?
                - Are there derivative metrics that could be calculated from others?

                Your response MUST follow this format:
                1. Overall Score: X/5
                2. Breakdown:
                - Specificity: X/5
                - Relevance: X/5
                - Non-Redundancy: X/5
                3. Issues Found:(optional)
                """

            # f"""You are a judge tasked with evaluating the quality of a table containing engineering metrics used to assess a {product}. 
            # Your goal is to rate the table on a scale from 1 to 10 based on the following criteria:
            #  * Specificity: Metrics should be clearly defined and quantifiable. Avoid vague or generic terms.
            #  * Relevance: Each metric should be meaningfully tied to the original product's goals, performance, or reliability.
            #  * Non-Redundancy: Metrics should be distinct and non-overlapping, avoiding repetitive or highly correlated entries.
            
            # Given the table of metrics below, provide a rating (1 = very poor, 10 = excellent), a brief rationale for your score, and point out any examples of vague, irrelevant, or redundant metrics if applicable. Be scrutinous.
            # {table}

            # IMPORTANT:
            # You **MUST** first output your evaluation, 
            # and secondly save the evaluation to a file by passing the **string** output as the first argument, and filename {filename} as the second argument, to the `save_judge_feedback` tool.
            # Thirdly, you **MUST** tell the user where you saved the evaluation.
            # """
        )


class RefineAgent(AssistantAgent):
        def __init__(self, model_client, similar_metrics):
            super().__init__(
                name="refine_agent",
                model_client=model_client,
                description=f"reduce redunduncy in similar metrics",
                system_message=f"""
                You are given the following list of similar metrics pairs for the associated customer needs. 
                1) If two metrics address the same need, reduce them down to only one metric for that need.
                2) If two metrics address two different needs, reduce them down to one metric for only one of the two needs that is most relevent.
                Important: do not change the wording of the customer needs.
                {similar_metrics}
                
                Output in the following format:
                    **Original redunduncies:**
                        {similar_metrics}
                    **CHANGES**
                        Output a list of the full metric and full need you finally chose for each pair. 
                """
            )



#---FOR LOOP----
async def my_run(my_agent, requirement_prompt):
    await Console(
            my_agent.on_messages_stream(
                    [TextMessage(content=requirement_prompt, source=my_agent.name)],
                    cancellation_token=CancellationToken(),
            )
        )


async def main():



    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process customer needs for product design')
    parser.add_argument('--model', type=str, help='Model client to use')
    parser.add_argument('--needs-file', type=str, help='Path to existing customer needs JSON file')
    parser.add_argument('--domain-file', type=str, help='Path to existing domain needs JSON file')
    parser.add_argument('--raw-final-json-filename', type=str, help='Path to existing raw final metrics JSON file')
    parser.add_argument('--final-json-filename', type=str, help='Path to existing final metrics JSON file')
    parser.add_argument('--result-folder', type=str, help='Path of the result folder')
    args = parser.parse_args()
    
    if args.result_folder:
        result_folder = args.result_folder
        os.makedirs(result_folder, exist_ok=True)
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = f"./results/{current_time}"
        os.makedirs(result_folder, exist_ok=True)

    stage1_skip, stage2_skip, stage3_skip = False, False, False
    if args.domain_file:
        stage1_skip = True
    if args.raw_final_json_filename:
        stage1_skip, stage2_skip = True, True
    if args.final_json_filename:
        stage1_skip, stage2_skip, stage3_skip = True, True, True

    if not args.model:
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
    elif args.model == "gpt4o":
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
    elif args.model == "claude3":
        model_client = AnthropicChatCompletionClient(model="claude-3-opus-20240229")
    elif args.model == "claude37":
        model_client = AnthropicChatCompletionClient(model="claude-3-7-sonnet-20250219")
    elif args.model == "llama31":
        model_client = OllamaChatCompletionClient(model="llama3.1")
    elif args.model == "gpt35":
        model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")
    else:
        raise ValueError(f"Invalid model client: {args.model}")

    #==============STAGE 1============== Get customer needs - either from elicitron API or existing file
    if not stage1_skip:
        print("\n==========STAGE 1==========\n")
        if args.needs_file:
            print(f"Using existing customer needs file: {args.needs_file}")
            try:
                with open(args.needs_file, 'r') as file:
                    needs_data = json.load(file)
            except FileNotFoundError:
                print(f"Error: File {args.needs_file} not found")
                return
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file {args.needs_file}")
                return
        else:
            print("Generating new customer needs using elicitron...")
            needs_data = await run_elicitron(result_folder)
            customer_needs_filename = f"cus_needs/{product.lower().replace(' ', '_')}_needs.json"
            with open(customer_needs_filename, 'w') as file:
                json.dump(needs_data, file, indent=4)
            print(f"Saved generated customer needs to: {customer_needs_filename}")

        # Format customer needs for processing
        customer_needs = '\n'.join([f"{need}" for need in needs_data['customer_needs']])
        print("\n==========customer_needs ready (STAGE 1 completed)==========\n", customer_needs)
    else:
        print("STAGE 1 skipped")



    #==============STAGE 2============== Run initial product agent to classify needs into domains
    if not stage2_skip:
        print("\n==========STAGE 2==========\n")
        if not args.domain_file:
            #initial product agent
            domain_needs_filename = f"domain_needs/{product.lower().replace(' ', '_')}_domain_needs.json"
            init_product_agent = InitialProductLeadAgent(model_client=model_client, result_file_path=domain_needs_filename)
            requirement_prompt = f"""Below is a list of customer needs related to {product}.
                    {customer_needs}"""
            print("-----------Starting initial product agent to classify needs into domains-----------")

            await Console(
                init_product_agent.on_messages_stream(
                        [TextMessage(content=requirement_prompt, source=init_product_agent.name)],
                        cancellation_token=CancellationToken(),
                )
            )
            # Read domain needs from the saved file
        else:
            domain_needs_filename = args.domain_file
            print(f"Using existing domain needs file: {domain_needs_filename}")
        
        with open(domain_needs_filename, 'r') as file:
            config = json.load(file) #domain needs saved to config, ready for stage 3
            print("\n==========domain needs ready (STAGE 2 completed)==========\n")
    else:
        print("STAGE 2 skipped")



    #==============STAGE 3==============Round Robin Group Chat for the five domain agents
    if not stage3_skip:
        print("\n==========STAGE 3: Round Robin Group Chat==========\n")
        if not args.raw_final_json_filename:
            roles_needs_dict = {}
            # Iterate over each role in the JSON and add it to the dictionary
            for role_data in config['roles']:
                role = role_data['role']
                needs = role_data['needs']
                roles_needs_dict[role] = needs
            
            all_domain_metrics = {} #each loop add a new domain metrics to the dictionary
            for agent_name in roles_needs_dict.keys():
                print(f"-----------Now {agent_name} is working-----------")
                json_filename = result_folder + "/" +f"{agent_name}_metrics.json"
                table_filename = result_folder + "/" + f"{agent_name}_table.png"
                my_agent = MetricsAgent(name=str(agent_name), model_client=model_client, domain=agent_name)
                supervisor_agent = SupervisorAgent(name=f'Supervisor_for_{str(agent_name)}', model_client=model_client, domain=agent_name, json_filename=json_filename)
                
                #round robin group chat
                my_agent_needs = '\n'.join(roles_needs_dict[agent_name])
                requirement_prompt = f"""Please provide domain specific metrics to address the following customer needs:
                            {my_agent_needs}"""
                termination = TextMentionTermination("SUBMIT") #termination condition
                team = RoundRobinGroupChat([supervisor_agent, my_agent], 
                                    max_turns=10,
                                    termination_condition=termination) #create a team of agents
                await Console(team.run_stream(task=requirement_prompt))   

                #save metrics to all_domain_metrics
                with open(json_filename, 'r') as file: #json already auto saved by the supervisor agent
                    table = json.load(file)
                    for customer_need, metrics in table.items():
                        if customer_need not in all_domain_metrics:
                            all_domain_metrics[customer_need] = metrics  # Save metrics for new customer needs
                        else:
                            all_domain_metrics[customer_need].extend(metrics)  # Append metrics for existing customer needs
                table = dict(table)
                save_table_as_image(table_filename, table) #convert to image for visualization

            print("-------Product lead agent is working on combining all the metrics into a final table------")
            raw_final_json_filename = result_folder + "/" + "raw_final_metrics.json"
            raw_final_table_filename = result_folder + "/" + "raw_final_table.png"
            product_lead_agent_init = ProductLeadAgent(name="product_lead_init", model_client=model_client, json_filename=raw_final_json_filename)
            requirement_prompt = f"""Please combine all the metrics into a final table and submit it. {all_domain_metrics}"""
    #         second_prompt = f"""
    # You are a product lead agent responsible for assembling a final table from the metrics provided. 

    # **IMPORTANT OUTPUT FORMAT:**  
    #         You **MUST** output a JSON object as a **string**, not a Python dictionary.  
    #         **DO NOT** add escape sequences (`\\n`, `\\"`), or unnecessary formatting.  
    #         The output **must** be a raw JSON string.
    #         DO NOT put the whole output on one line.

    #         **VALID EXAMPLE (Correct Format):**
    #         ```json
    #         "{{
    #             \\"Need1\\": [
    #                 [\\"Metric1\\", \\"Unit1\\"],
    #                 [\\"Metric2\\", \\"Unit 2\\"]
    #             ],
    #             \\"Need 2\\": [
    #                 [\\"Metric 1\\", \\"Unit 1\\"],
    #                 [\\"Metric 2\\", \\"Unit 2\\"]
    #             ]
    #         }}"
    #         ```

    # Pass EXACTLY {raw_final_json_filename} as the first argument, and this JSON **string** as the second argument to the `save_table` tool.
    # Do NOT ask for feedback or invoke the judge, just simply call the function with the correct inputs.
    # """
    #         product_lead_agent_init._system_messages[0].content = second_prompt

            await Console(
                    product_lead_agent_init.on_messages_stream(
                            [TextMessage(content=requirement_prompt, source=product_lead_agent_init.name)],
                            cancellation_token=CancellationToken(),
                    )
                ) #json already auto saved 
            
            with open(raw_final_json_filename, 'r') as file:
                table = json.load(file)
            table_dict = dict(table)
            save_table_as_image(raw_final_table_filename, table_dict)
        else:
            raw_final_json_filename = args.raw_final_json_filename
            with open(raw_final_json_filename, 'r') as file:
                table = json.load(file)
        
        metrics, units = visualizer.load_metrics(raw_final_json_filename)
        print(f"Loaded {len(metrics)} unique metrics")
        print("\n==========initial table ready (STAGE 3 completed)==========\n")
    else:
        print("STAGE 3 skipped")




    #==============STAGE 4============== Refining the final table with Embedding and Refine Agent
    print("\n==========STAGE 4: Refining the final table with Embedding and Refine Agent==========\n")
    if not args.final_json_filename:
        final_json_filename = result_folder + "/" + "final_metrics.json"
        final_table_filename = result_folder + "/" + "final_table.png"

        # Compute embeddings
        print("Computing embeddings...")
        visualizer.compute_embeddings()
        # Find similar pairs
        print("\nFinding similar metrics...")
        similarity_threshold = 0.6
        similar_pairs = visualizer.find_similar_pairs(similarity_threshold=similarity_threshold)
        # Visualization
        print("\nGenerating visualization...")
        fig = visualizer.visualize()
        html_visualization_name = result_folder + "/" + "metrics_similarity_visualization.html"
        fig.write_html(html_visualization_name)
        fig.show()
        with redirect_stdout(f):
            visualizer.print_similar_pairs(similar_pairs, similarity_threshold=similarity_threshold)
        redundunt_pair_descriptions = f.getvalue()

        #group chat for product lead and refine
        refine_agent = RefineAgent(model_client=model_client, similar_metrics=redundunt_pair_descriptions)
        product_lead_agent = ProductLeadAgent(name="product_lead", model_client=model_client, json_filename=final_json_filename)
        requirement_prompt = f"""Please combine and optimize all domain metrics into a final table to reduce redunduncy. Note that the customer needs wording are not allowed to be changed. {table}"""
        termination = TextMentionTermination("SUBMIT") #termination condition
        team = RoundRobinGroupChat([product_lead_agent, refine_agent], 
                        max_turns=8,
                        termination_condition=termination) #create a team of agents

        await Console(team.run_stream(task=requirement_prompt))
        with open(final_json_filename, 'r') as file:
            table = json.load(file)
        table_dict = dict(table)
        save_table_as_image(final_table_filename, table_dict)
    else:
        final_json_filename = args.final_json_filename
        with open(final_json_filename, 'r') as file:
            table = json.load(file)

    print("\n==========Final table ready to be judged(STAGE 4 completed)==========\n")
    
    
    #==============STAGE 5============== Judge Agent
    print("\n==========STAGE 5: Judge Agent==========\n")
    judge_agent = LLM_as_Judge(name="judge", model_client=model_client)
    requirement_prompt = f"Provide an evaluation of the following engineering metrics {table}"

    response = await Console(
            judge_agent.on_messages_stream(
                    [TextMessage(content=requirement_prompt, source=judge_agent.name)],
                    cancellation_token=CancellationToken(),
            )
        )
    
    # Extract and save scores
    extract_and_save_scores(response, result_folder)
    

if __name__ == "__main__":
    asyncio.run(main())