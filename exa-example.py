# Example file to give context to LLM on how to use Exa API

from exa_py import Exa
import os

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

result = exa.answer("consumer AI startups that had breakthrough recently", text=True)
# text = false if we don't need to return webpage text for every citation as given in example below

print(result)

# example response
result = {
    "answer": "As of February 26, 2025, notable consumer AI startups that have recently achieved breakthroughs include **DeepSeek**, a Chinese startup that has gained attention for its cost-efficient AI model, which has significantly impacted the tech market since its inception just a year ago ([CNN](https://www.cnn.com/2025/01/27/tech/deepseek-ai-explainer/index.html), [US Funds](https://www.usfunds.com/resource/chinas-ai-breakthrough-sends-nvidia-reeling-and-sparks-national-security-fears/)). \n\nAdditionally, **Komodo Health** has made strides in the healthcare sector with its innovative MapAI platform, which processes over 330 million health data points to enhance healthcare delivery ([Solute Labs](https://www.solutelabs.com/blog/top-ai-healthcare-startups)). These advancements reflect a growing trend of AI startups making significant impacts across various industries.",
    "citations": [
        {
            "id": "https://www.cnn.com/2025/01/27/tech/deepseek-ai-explainer/index.html",
            "title": "What is DeepSeek, the Chinese AI startup that shook the tech world?",
            "url": "https://www.cnn.com/2025/01/27/tech/deepseek-ai-explainer/index.html",
            "publishedDate": "2025-01-27T12:00:00.000Z",
            "author": "null",
            "text": "The new AI model was developed by DeepSeek, a startup that was born just a year ago and has somehow managed a breakthrough that famed tech ...",
        },
        {
            "id": "https://www.solutelabs.com/blog/top-ai-healthcare-startups",
            "title": "Breakthrough AI Startups Making Waves in Healthcare in 2025",
            "url": "https://www.solutelabs.com/blog/top-ai-healthcare-startups",
            "publishedDate": "2025-01-27T12:00:00.000Z",
            "author": "null",
            "text": "Komodo Health has emerged as one of the pioneering healthcare AI companies with its revolutionary MapAI platform that processes over 330 million ...",
        },
        {
            "id": "https://www.usfunds.com/resource/chinas-ai-breakthrough-sends-nvidia-reeling-and-sparks-national-security-fears/",
            "title": "China's AI Breakthrough Sends NVIDIA Reeling and Sparks ...",
            "url": "https://www.usfunds.com/resource/chinas-ai-breakthrough-sends-nvidia-reeling-and-sparks-national-security-fears/",
            "publishedDate": "2025-01-31T12:00:00.000Z",
            "author": "null",
            "text": "Enter DeepSeek, a Chinese AI startup that's sent shockwaves through the market with the release of a new, highly cost-efficient AI model.",
        },
    ],
}
