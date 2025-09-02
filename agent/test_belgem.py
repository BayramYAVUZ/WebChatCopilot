import os
from dotenv import load_dotenv
import google.generativeai as genai # type: ignore


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Merhaba! bana 3 renk söyle")
print(response.text)
