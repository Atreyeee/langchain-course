from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os

load_dotenv()
def main():
    print("Hello")
    information="""
    Jude Victor William Bellingham (born 29 June 2003) is an English professional footballer who plays as an attacking midfielder for La Liga club Real Madrid and the England national team. Widely regarded as one of the best players in the world, he is known for his dribbling, finishing and ball-winning ability.[4][5][6][7]

    Bellingham joined Birmingham City as an under-8, became the club's youngest first-team player when he made his senior debut in August 2019 at the age of 16, and played regularly during the 2019–20 season. He signed with Borussia Dortmund in July 2020, and in his first appearance became the club's youngest goal-scorer. Over three seasons with the club he made 132 appearances and won the 2020–21 DFB-Pokal; his performances in the 2022–23 season helped Dortmund finish as runners-up and earned him the Bundesliga Player of the Season award. Later that year, he won the European Golden Boy and the Kopa Trophy.

    Bellingham signed for Real Madrid in 2023 for €103 million. In his first season, he was the club's top league scorer, helping them to the league title and Champions League, and was voted La Liga Player of the Season. He was included in the FIFPRO World 11 for three consecutive years (2023, 2024, 2025), and finished third in voting for the Ballon d'Or and The Best FIFA Men's Player in 2024.

    Bellingham represented England at the under-15, under-16, under-17, and under-21 levels. He made his first appearance for the senior team in November 2020, and was part of the squads for the UEFA European Championship in 2020 and 2024, as well as the FIFA World Cup in 2022 and 2026. At the 2026 World Cup, Bellingham helped England finish in third place as he scored seven goals, making him England's all-time highest goal-scorer in a single World Cup tournament.[8]
    """
    summary_template="""given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them"""
    summary_prompt_template=PromptTemplate(
        input_variables=["information"],template=summary_template
    )
    #llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0)
    llm=ChatOllama(model="gemma3:270m",temperature=0)
    chain= summary_prompt_template | llm
    response=chain.invoke({"information":information})
    print(response.content)
if __name__=="__main__":
    main()

#uv run python -u main.py
