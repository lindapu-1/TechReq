import uvicorn
from fastapi import FastAPI, HTTPException, Body
from starlette.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import msgpack
from .config import model
from openai import OpenAI
import instructor
from .stakeholders import (
    StakeholderBase,
    Stakeholder,
    Stakeholders,
    interview_stakeholder,
    top_n_diverse_stakeholders,
    Interview,
)
from .latent_needs import LatentNeeds
import uuid
from datetime import datetime


app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
client = OpenAI(api_key = "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")
client = instructor.patch(OpenAI())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


class StakeholderCreateaRequestParallel(BaseModel):
    design_context: str
    model_steering: str


class StakeholderCreateaRequestSerial(BaseModel):
    design_context: str
    model_steering: str
    number_of_stakeholders: int

class CustomerNeedsRequest(BaseModel):
    product: str
    report: str

class CustomerNeedsList(BaseModel):
    product: str
    customer_needs: List[str] = Field(..., description="List of clear, concise customer needs")

def stakeholder_creator_parallel(request: StakeholderCreateaRequestParallel):
    model_steering = request.model_steering
    if model_steering != "":
        model_steering = f"{request.model_steering}."

    stakeholder_creator_prompt = (
        f"You're a user creator for a new design of a {request.design_context}. "
        f"The users must be diverse consumers of the product, whom you will interview afterwards to identify user needs. "
        f"{model_steering} "
    )
    response = client.chat.completions.create(
        model=model,
        response_model=StakeholderBase,
        messages=[
            {"role": "user", "content": stakeholder_creator_prompt},
        ],
        max_tokens=1024,
        temperature=1.0,
    )
    return response


def stakeholder_creator_serial(request: StakeholderCreateaRequestSerial):
    logger.debug(f"Creating {request.number_of_stakeholders} users.")

    class StakeholderList(BaseModel):
        stakeholders: List[StakeholderBase] = Field(
            default=[],
            description=f"List of users, should be of length {request.number_of_stakeholders}",
        )

    model_steering = request.model_steering
    if model_steering != "":
        model_steering = f"You MUST consider the following: {request.model_steering}."

    stakeholder_creator_prompt = f"You're a user creator for a new design of {request.design_context}. {model_steering} The users must be diverse consumers of the product, whom you will interview afterwards to identify user needs. Please create a list of {request.number_of_stakeholders} users."

    response = client.chat.completions.create(
        model=model,
        response_model=StakeholderList,
        messages=[
            {"role": "user", "content": stakeholder_creator_prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
    )
    return response


@app.post("/create_stakeholder_parallel")
def create_stakeholder_parallel(request: StakeholderCreateaRequestParallel):
    print("=========create_stakeholder_parallel is getting called=========")
    created_stakeholder = stakeholder_creator_parallel(request)
    return created_stakeholder


@app.post("/create_stakeholder_serial")
def create_stakeholder_serial(request: StakeholderCreateaRequestSerial):
    created_stakeholder = stakeholder_creator_serial(request)
    return created_stakeholder


class InterviewRequest(BaseModel):
    stakeholder: Stakeholder
    design_context: str
    questions: List[str]


@app.post("/conduct_interview")
async def conduct_interview(request: InterviewRequest):
    stakeholder = request.stakeholder
    design_context = request.design_context
    questions = request.questions
    interview = interview_stakeholder(stakeholder, questions, design_context)
    return interview


class TopNDiverseRequest(BaseModel):
    stakeholders: Stakeholders
    n: int


@app.post("/filter_top_diverse")
def filter_top_diverse(request: TopNDiverseRequest):
    logger.debug(f"Received {len(request.stakeholders)} stakeholders.")
    logger.debug(f"Selecting top {request.n} diverse stakeholders.")
    logger.debug(f"Stakeholders: {request.stakeholders}")

    filtered_stakeholders = top_n_diverse_stakeholders(request.stakeholders, request.n)
    return filtered_stakeholders


class ReportRequest(BaseModel):
    interviews: list
    questions: list


@app.post("/create_report")
def create_report(request: ReportRequest):
    print("=========create_report is getting called=========")
    interviews = request.interviews
    questions = request.questions

    interview_text = "Interviews Summary:\n"
    for i, interview in enumerate(interviews):
        logger.debug(f"Interview {i+1}: {interview}")
        interview_text += f"Interview {i+1}:\n"
        interview_text += f"User Type: {interview['stakeholder_type']}\n"
        interview_text += (
            f"User Characteristics: {interview['personality_description']}\n"
        )

    questions_text = "Questions:\n"
    for question in questions:
        questions_text += question + "\n"

    report_prompt = (
        "Create a report summarizing the key insights and requirements from the following interview questions and answers:\n\n"
        f"{questions_text}\n"
        f"{interview_text}\n"
        "The report should focus exclusively on the information provided in these interview questions and answers. Your task is to distill the critical points and requirements without directly copying any part of the interview or question text. Ensure that the report maintains the essence of the interviews and questions while presenting the information succinctly and professionally. Do not include information that requires placeholder text."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": report_prompt}],
        max_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content


class LatentNeedIdentificationRequest(BaseModel):
    design_context: str
    interview: Interview


@app.post("/latent_need_identification")
def latent_need_identification(request: LatentNeedIdentificationRequest):
    logger.debug(f"Identifying latent needs for {request}.")
    design_context = request.design_context
    QandAs = request.interview.QandAs

    latent_need_identification_prompt = (
        f"Based on the following interview, identify the latent needs for the design process of {design_context}:\n\n"
        f"{QandAs}\n\n"
    )
    response = client.chat.completions.create(
        model=model,
        response_model=LatentNeeds,
        messages=[
            {"role": "user", "content": latent_need_identification_prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
    )
    return response


@app.post("/extract_customer_needs")
def extract_customer_needs(request: CustomerNeedsRequest):
    print("=========This is getting called=========")
    logger.debug(f"Extracting customer needs for {request.product}")
    
    prompt = f"""
Based on the following report about {request.product}, extract and standardize the customer needs.

Report:
{request.report}

Create a comprehensive list of clear, specific customer needs following these guidelines:
1. Each need should be a complete, specific statement with clear requirements
2. Use consistent format: "The {request.product} [verb] [specific requirement]"
3. Focus on what the product must do, not how it might do it
4. Be specific and measurable where possible
5. Avoid duplicate needs
6. Exclude general background information or context
7. Consider needs from multiple perspectives:
   - Safety and reliability
   - Performance and functionality
   - User experience and convenience
   - Environmental impact
   - Cost and maintenance
   - Regulatory compliance
   - Aesthetics and design
   - Durability and longevity

Format your response as a list of customer needs similar to these examples:
"The suspension reduces vibration to the hands during high-speed descents"
"The suspension allows easy traversal of slow, difficult terrain while maintaining stability"
"The suspension enables high-speed descents on bumpy trails without compromising control"
"The suspension maintains consistent performance across various weather conditions"
"The suspension requires minimal maintenance over extended periods of use"

Important:
- Extract major needs mentioned in the report
- Make each need specific and measurable (include numbers where possible)
- Break down complex features into separate, specific needs
- Use precise language (avoid vague terms like "adequate", "superior", "advanced")
- Include specific performance requirements and safety thresholds
- Focus on concrete, testable requirements rather than general features
- If a feature has multiple aspects, list each aspect as a separate need
"""

    response = client.chat.completions.create(
        model=model,
        response_model=CustomerNeedsList,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    
    return response.model_dump()

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
