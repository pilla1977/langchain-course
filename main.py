from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():

    information = """
    Jesse Louis Jackson (né Burns; October 8, 1941 – February 17, 2026) was an American civil rights activist, LGBTQ rights activist, politician, and ordained Baptist minister. A protégé of Martin Luther King Jr. and James Bevel during the civil rights movement, he became one of the most prominent civil rights leaders of the late 20th and early 21st centuries, and an ardent and early supporter of LGBTQ rights in the United States. From 1991 to 1997, he served as a shadow delegate and shadow senator for the District of Columbia.
    Born in Greenville, South Carolina, Jackson began his activism in the 1960s and founded the organizations that later merged to form the Rainbow/PUSH Coalition. Expanding his work into international affairs in the 1980s, he became a vocal critic of the Reagan administration and launched a presidential campaign in 1984. Initially viewed as a fringe candidate, he finished third for the Democratic nomination behind former vice president Walter Mondale and Senator Gary Hart. He continued his activism and mounted a second presidential bid in 1988, finishing as the runner‑up for the Democratic nomination to Massachusetts governor Michael Dukakis.
    Jackson did not seek the presidency again, but in 1990 he was elected as the District of Columbia's shadow senator, serving one term during the Bush and Clinton administrations. Although initially critical of President Bill Clinton, he later became a supporter. Jackson hosted Both Sides with Jesse Jackson on CNN from 1992 to 2000. A critic of police brutality, the Republican Party, and conservative policies, he was widely regarded as one of the most influential African‑American activists of his era.
    """
    summary_template = """
    Given the information {information} about a person, i want you to create:
    1) A short summary.
    2) Give me only two interesting facts about them.
    """

    summary_prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
