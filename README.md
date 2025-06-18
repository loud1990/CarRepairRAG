# Car Repair RAG Chatbot Setup Guide

This guide will walk you through setting up and running the Car Repair RAG Chatbot.

## Prerequisites

Before you begin, ensure you have Python installed on your system.

## Setup Instructions

Follow these steps to get the chatbot up and running:

### Step 1: Configure your OpenAI API Key

Create a file named `.env` in the root directory of your project. Add your OpenAI API key to this file in the following format:

```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

### Step 2: Install Dependencies

Open your terminal or command prompt, navigate to the root directory of this project, and run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Chatbot

After installing the dependencies, you can start the chatbot by executing the following command in your terminal:

```bash
python chatbot.py
```

The chatbot should now be running and ready to use.

### Notes

At any time, you may type `quit` and hit enter to exit the chatbot.