
import json
import os
from typing import List, Optional
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
)
try:
    from langchain.output_parsers import OutputFixingParser
except ImportError:
    from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Add it to a .env file in D:\\Task 3."
    )

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_retries=2,
    timeout=30,
)


# 1. BEFORE: raw reply that json.loads() cannot parse
print("\n" + "=" * 72)
print("1. BEFORE — RAW MODEL REPLY / json.loads() FAILURE")
print("=" * 72)

raw_reply = (
    "Here is the movie information you requested:\n"
    "```json\n"
    '{"title": "Interstellar", "year": 2014}\n'
    "```"
)
print("Raw model-style reply:\n", raw_reply)

try:
    json.loads(raw_reply)
except json.JSONDecodeError as exc:
    print("\nEXPECTED ERROR:", type(exc).__name__)
    print("json.loads() failed because the reply includes friendly text and Markdown fences.")
    print("Error details:", exc)


# 2. Pydantic schema
class ReviewInfo(BaseModel):
    reviewer: str = Field(description="Name or role of the reviewer")
    sentiment: str = Field(description="Short sentiment label, such as positive")


class MovieInfo(BaseModel):
    title: str = Field(description="Official title of the movie")
    year: int = Field(ge=1888, le=2100, description="Release year, from 1888 to 2100")
    rating: float = Field(ge=0, le=10, description="Movie rating from 0 to 10")
    genres: List[str] = Field(description="List of movie genres")
    summary: str = Field(description="A brief summary of the movie")
    director: Optional[str] = Field(default=None, description="Director name, if known")
    review: ReviewInfo = Field(description="Nested review with reviewer and sentiment")


# 3. AFTER: with_structured_output on three inputs
print("\n" + "=" * 72)
print("2. AFTER — STRUCTURED OUTPUT ON THREE INPUTS")
print("=" * 72)

movie_inputs = [
    "Interstellar (2014)",
    "WALL-E (2008)",
    "The Martian (2015)",
]

# Tool-calling structured output for OpenRouter compatibility
try:
    structured_llm = llm.with_structured_output(MovieInfo)
except Exception as exc:
    print("Could not configure structured output:", type(exc).__name__, exc)
    structured_llm = None

if structured_llm is not None:
    for index, movie in enumerate(movie_inputs, start=1):
        print(f"\n--- Structured input {index}: {movie} ---")
        question = (
            f"Return information about {movie}. Include title, year, rating from 0 to 10, "
            "genres as a list, a brief summary, director if known (otherwise null), and "
            "a nested review with reviewer='General audience' and a sentiment label. "
            "Return only a JSON object matching the schema."
        )
        try:
            result = structured_llm.invoke(question)
            print("Returned object:", result)
            print("Returned type:", type(result))
            print("Direct field access — result.title:", result.title)
            print("Direct field access — result.rating:", result.rating)
        except Exception as exc:
            print("STRUCTURED OUTPUT ERROR:", type(exc).__name__, str(exc))
            print("This may indicate provider/model JSON-mode support or API limits.")


# 4. TypedDict version, with Annotated descriptions
class ReviewDict(TypedDict):
    reviewer: Annotated[str, "Name or role of the reviewer"]
    sentiment: Annotated[str, "Short sentiment label, such as positive"]


class MovieInfoDict(TypedDict):
    title: Annotated[str, "Official title of the movie"]
    year: Annotated[int, "Release year"]
    rating: Annotated[float, "Movie rating from 0 to 10"]
    genres: Annotated[List[str], "List of movie genres"]
    summary: Annotated[str, "Brief movie summary"]
    director: Annotated[Optional[str], "Director name, if known"]
    review: Annotated[ReviewDict, "Nested review information"]


print("\n" + "=" * 72)
print("3. TYPEDDICT VERSION")
print("=" * 72)
try:
    typed_llm = llm.with_structured_output(MovieInfoDict)
    typed_result = typed_llm.invoke(
        "Return a JSON object about Finding Nemo (2003), including title, year, "
        "rating from 0 to 10, genres list, summary, director if known, and nested "
        "review with reviewer='General audience' and sentiment='positive'."
    )
    print("TypedDict result:", typed_result)
    print("Returned type:", type(typed_result))
except Exception as exc:
    print("TYPEDDICT OUTPUT ERROR:", type(exc).__name__, str(exc))

print("\nTypedDict vs Pydantic:")
print("• TypedDict describes the expected dictionary shape and supports type hints.")
print("• It is lightweight but does not validate values at runtime by itself.")
print("• Pydantic validates data and constraints, such as a rating from 0 to 10.")
print("• Pydantic provides detailed validation errors; TypedDict is useful for plain dictionaries.")


# 5. Validation error
print("\n" + "=" * 72)
print("4. PYDANTIC VALIDATION ERROR")
print("=" * 72)

bad_movie = {
    "title": "Impossible Rating",
    "year": 2200,  # violates le=2100
    "rating": 15.5,  # violates le=10
    "genres": ["Drama"],
    "summary": "A deliberately invalid example for the assignment.",
    "director": None,
    "review": {"reviewer": "Test", "sentiment": "unknown"},
}

try:
    MovieInfo.model_validate(bad_movie)
except ValidationError as exc:
    print("EXPECTED ValidationError caught:")
    print(exc)

print("\nType-valid vs factually correct:")
print("Type-valid means the data has the required fields and acceptable types and passes declared constraints.")
print("Factually correct means the information matches reality. An object can pass validation but still contain a wrong director, year, or summary.")


# 6. PydanticOutputParser + get_format_instructions()
print("\n" + "=" * 72)
print("5. PYDANTIC OUTPUT PARSER")
print("=" * 72)

parser = PydanticOutputParser(pydantic_object=MovieInfo)
format_instructions = parser.get_format_instructions()
print("get_format_instructions():\n", format_instructions)

parser_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a careful movie-information assistant. Follow these formatting "
        "instructions exactly:\n{format_instructions}",
    ),
    ("human", "{question}"),
])

parser_chain = parser_prompt | llm | parser
try:
    parsed_movie = parser_chain.invoke({
        "format_instructions": format_instructions,
        "question": (
            "Create a MovieInfo object for Inception (2010). Use rating from 0 to 10, "
            "include genres, summary, director if known, and a nested review."
        ),
    })
    print("Parsed Pydantic object:", parsed_movie)
    print("Parsed type:", type(parsed_movie))
    print("Direct field access — parsed_movie.title:", parsed_movie.title)
except Exception as exc:
    print("PYDANTIC PARSER ERROR:", type(exc).__name__, str(exc))


# 7. OutputFixingParser: repair deliberately broken JSON
print("\n" + "=" * 72)
print("6. OUTPUT FIXING PARSER — BROKEN JSON -> REPAIRED OBJECT")
print("=" * 72)

fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=MovieInfo),
    llm=llm,
)

broken_json = (
    '{"title": "Arrival", "year": 2016, "rating": 7.9, '
    '"genres": ["Sci-Fi", "Drama"], "summary": "A linguist studies alien language", '
    '"director": "Denis Villeneuve", '
    '"review": {"reviewer": "General audience", "sentiment": "positive",}}'
)
print("Deliberately broken JSON (trailing comma):\n", broken_json)

try:
    json.loads(broken_json)
except json.JSONDecodeError as exc:
    print("\nEXPECTED JSON ERROR before fixing:", type(exc).__name__, exc)

try:
    fixed_result = fixing_parser.parse(broken_json)
    print("\nREPAIRED AND PARSED OUTPUT:")
    print(fixed_result)
    print("Returned type:", type(fixed_result))
    print("Repaired field access — fixed_result.title:", fixed_result.title)
except Exception as exc:
    print("\nOUTPUT FIXING ERROR:", type(exc).__name__, str(exc))
    print("The fixing parser asks the model to repair malformed output. Provider/model behavior can vary.")


# 8. CommaSeparatedListOutputParser
print("\n" + "=" * 72)
print("7. COMMA-SEPARATED LIST OUTPUT PARSER")
print("=" * 72)

list_parser = CommaSeparatedListOutputParser()
list_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Give exactly five concise movie-analysis keywords separated by commas. "
        "Do not use bullets or extra explanation.",
    ),
    ("human", "Give keywords for this answer: {answer}"),
])

list_chain = list_prompt | llm | list_parser
try:
    keywords = list_chain.invoke({
        "answer": (
            "Interstellar is a science-fiction film about space exploration, "
            "relativity, family, survival, and humanity's future."
        )
    })
    print("Python keyword list:", keywords)
    print("Returned type:", type(keywords))
except Exception as exc:
    print("LIST PARSER ERROR:", type(exc).__name__, str(exc))

print("\n" + "=" * 72)
print("TASK 3 SCRIPT FINISHED")
print("=" * 72)
