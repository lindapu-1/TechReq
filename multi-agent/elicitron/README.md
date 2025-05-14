# SIMULATING REQUIREMENTS ELICITATION WITH LLM AGENTS
<p align="center">
  <img src="frontend/src/static/logo.png" alt="" width="300">
</p>

## Intro
This repository hosts the code for our paper on using multiple LLM agents to simulat latent design elicitation. Our approach involves LLM agents collaborating through `chain-of-thought reasoning' to enhance early stages of mechanical system design, providing nuanced, context-aware insights and innovative solutions.



## Project Structure

The project is divided into two main parts:
- `backend`: This contains the FastAPI server code.
- `frontend`: This is a React application that interacts with the backend.


## Getting Started

### Prerequisites

- Node.js and npm (https://nodejs.org/)
- Python 3.10+ (https://www.python.org/)


### Installation

Frontend:

```
cd ../frontend
npm install

```
Backend:

```
cd ../backend
pip install -r requirements.txt
```

## Running the Application

### Frontend

```
cd ../frontend
npm start
```

### Backend

```
cd ../backend
uvicorn backend:app --reload
```

### Architecture

The frontend and backend are designed to run independently. The frontend is a React application that interacts with the backend through a REST API. The backend is a FastAPI server that serves the frontend and provides the API endpoints.

Architecture for Requirements Elicitation Using LLMs is shown below. First, LLM agents are generated within a design context in either serial and parallel fashion (incorporating diversity sampling to represent varied user perspectives). These agents then engage in simulated product experience scenarios, documenting each step (Action, Observation, Challenge) in detail. Following this, they undergo an agent interview process, where questions are asked and answered to identify latent user needs. A report is generated if latent needs are identified through a zero-shot scoring mechanism, which then informs the design requirements.

<p align="center">
  <img src="frontend/src/static/architecture.png" alt="" width="900">
</p>
