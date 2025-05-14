import uuid
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .config import model
from .chain_of_thoughts import determine_stakeholder, description_for_product_experience
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = instructor.patch(OpenAI())


class StakeholderBase(BaseModel):
    chain_of_thought: str = Field(..., description=determine_stakeholder)
    stakeholder_type: str = Field(..., description="Type of user (five words max)")
    personality_description: str = Field(
        ...,
        description="Characteristics description of the user (be specific and detailed))",
    )


class Stakeholder(StakeholderBase):
    id: str


Stakeholders = List[Stakeholder]


class QandA(BaseModel):
    question: str
    answer: str


class Interview(BaseModel):
    id: str
    stakeholder_type: str
    personality_description: str
    product_usage_experience: str
    QandAs: List[QandA]


class ProductExperience(BaseModel):
    product_usage_experience: str = Field(
        ..., description=description_for_product_experience
    )


class SimpleQuestionResponse(BaseModel):
    answer: str


def interview_stakeholder(
    stakeholder: Stakeholder, questions: list, design_context: str
):
    # Function to generate product usage experience
    def generate_product_experience(stakeholder):
        prompt_template = (
            "As a '{stakeholder_type}', characterized by '{personality_description}', "
            "you're being interviewed to identify user needs for a new design of '{design_context}'. "
        )
        experience_prompt = prompt_template.format(
            stakeholder_type=stakeholder.stakeholder_type,
            personality_description=stakeholder.personality_description,
            #description_for_product_experience=description_for_product_experience,
            design_context=design_context,
        )
        experience_response = client.chat.completions.create(
            model=model,
            response_model=ProductExperience,
            messages=[{"role": "user", "content": experience_prompt}],
            max_tokens=4096,
            temperature=1.0,
        )

        return experience_response.product_usage_experience

    product_experience = generate_product_experience(stakeholder)
    q_and_a_list = []

    interview_history = {}

    for question in questions:
        # interview_prompt = (
        #     "Reflecting on your shared product experience:\n\n"
        #     f"{product_experience}\n\n"
        #     "Considering the specifics of your experience, including any challenges encountered, "
        # )

        interview_prompt = (
            "You are now being asked to answer a set of interview questions. "
            "When answering, remember to reflect back on the product experience you shared:\n\n"
            f"{product_experience}\n\n"
        )

        if interview_history:
            interview_prompt += "Here's a recap of our previous discussion:\n"
            for prev_question, prev_answer in interview_history.items():
                interview_prompt += f"- **Q:** {prev_question}\n  **A:** {prev_answer}\n"
            interview_prompt += "\nPlease keep this context in mind and avoid repetitive information. \n\n"

        interview_prompt += f"Now, kindly answer the following question. "
        interview_prompt += f"Be concise and specific to the challenges you encountered.\n\n"

        interview_prompt += f"**Current Question:** {question}\n"

        response = client.chat.completions.create(
            model=model,
            response_model=SimpleQuestionResponse,
            messages=[{"role": "user", "content": interview_prompt}],
            max_tokens=4096,
            temperature=1.0,
        )
        q_and_a_list.append(QandA(question=question, answer=response.answer))

        interview_history[question] = response.answer

        interview_instance = Interview(
            id=stakeholder.id,
            stakeholder_type=stakeholder.stakeholder_type,
            personality_description=stakeholder.personality_description,
            product_usage_experience=product_experience,
            QandAs=q_and_a_list,
        )

    logger.debug(
        f"interview_instance for {stakeholder.stakeholder_type}: {interview_instance}"
    )

    return interview_instance


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def assign_embeddings(stakeholders: Stakeholders) -> list:
    logger.debug(f"Number of stakeholders received: {len(stakeholders)}")
    stakeholder_embeddings = []
    for s in stakeholders:
        stakeholder_embeddings.append(get_embedding(s.personality_description))
    return stakeholder_embeddings


def top_n_diverse_stakeholders(stakeholders: Stakeholders, n: int) -> List[Stakeholder]:
    if n >= len(stakeholders):
        return stakeholders

    logger.debug(f"Selecting top {n} diverse stakeholders.")
    stakeholder_embeddings = assign_embeddings(stakeholders)

    matrix = np.vstack(stakeholder_embeddings)
    kmeans = KMeans(n, init='k-means++', random_state=42)
    kmeans.fit(matrix)

    top_stakeholders = [None] * n
    logger.debug(f"Kmeans {kmeans.labels_}")
    for stakeholder, category in zip(stakeholders, kmeans.labels_):
        if top_stakeholders[category] == None: 
            top_stakeholders[category] = stakeholder
            # randomly choose one from the category
    
    # Calculate the average embedding
    #average_embedding = np.mean(stakeholder_embeddings, axis=0)

    # Calculate diversity scores
    #diversity_scores = []
    #for stakeholder, embedding in zip(stakeholders, stakeholder_embeddings):
    #    personality_score = np.linalg.norm(embedding - average_embedding)
    #    diversity_scores.append((stakeholder, personality_score))

    # Sort stakeholders by diversity score
    #sorted_stakeholders = sorted(diversity_scores, key=lambda x: x[1], reverse=True)[:n]
    #top_stakeholders = [stakeholder for stakeholder, score in sorted_stakeholders]

    logger.debug(f"Selected {len(top_stakeholders)} top diverse stakeholders.")
    return top_stakeholders
