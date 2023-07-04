import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

# Load environment variables from .env file
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.7, "max_new_tokens":4000})


template = """
Question:Resumir en español, el siguiente texto en 100 palabras: {question}
Answer: resumen.


"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = """ 
¿Cuánto deben ocupar las pantallas en los colegios? ¿Su uso está afectando a la adquisición de competencias consideradas esenciales? ¿Se crean diferencias entre centros? ¿Los datos conseguidos con un medio digital son fiables? Desde hace algunos meses, este es el debate en Suecia. El lugar que deben ocupar las pantallas y la tecnología digital en las escuelas, incluido el tiempo de exposición, algo que ha sido cuestionado por los profesionales de la salud. Y va en línea con la decisión que adoptó el Ayuntamiento de Barcelona para sus escuelas infantiles, eliminando los aparatos tecnológicos de las aulas.

Fruto de ese debate, la nueva ministra de Educación, Lotta Edholm, ha aplazado la estrategia de la Agencia Nacional de Educación Escolar (Skolverket) en su plan digital, que fue presentado en diciembre de 2022. Entonces, la ministra ya escribió en un artículo contra la “actitud acrítica que considera la digitalización como algo bueno, independientemente de su contenido”, lo que lleva a “dejar de lado” los libros de texto, que, según señaló, tienen “ventajas que ninguna tableta puede sustituir”.
La ministra no niega el aprendizaje de la competencia digital de los niños, no elimina las pantallas, pero no va a invertir más en la tecnología y pone el énfasis en el papel. “El informe Pirls (sobre comprensión lectora) es una señal de que tenemos una crisis de lectura en las escuelas suecas. En el futuro, el Gobierno quiere ver más libros de texto y menos tiempo de pantalla en la escuela”.
Suecia, un país de 10 millones de habitantes, obtuvo una puntuación en el Informe Pirls 2021 de 544, situándose por encima de la media europea (528), la española (522) y la catalana (507). Pero esta cifra ha caído 11 puntos respecto al informe de 2016.

Según avanzó, el hecho de que los alumnos aprendan a leer y comprendan lo que leen es un requisito previo para el aprendizaje global y en los centros educativos se está perdiendo el foco en este objetivo. En su opinión, es preocupante que la capacidad de lectura esté disminuyendo entre los niños y los jóvenes por lo que las escuelas suecas deben volver a lo básico. “Hay que centrarse en competencias básicas como la lectura, la escritura y el cálculo”, afirmó recientemente Edholm según algunos diarios suecos.
Para remediar la situación, el Gobierno de centro-derecha anunció el pasado 15 de mayo que desbloqueará 685 millones de coronas (60 millones de euros) este año y 500 millones (44 millones de euros) anuales en 2024 y 2025, para acelerar el regreso de los libros de texto a las escuelas. “Esto forma parte del retorno de la lectura a la escuela, en detrimento del tiempo de pantalla”, advirtió la ministra. El objetivo es garantizar un libro por alumno y por asignatura. Asimismo, el gobierno ha invertido partidas económicas específicas destinadas a la compra de material escolar didáctico y, en concreto, el equivalente a unos 4 millones de euros a reforzar el desarrollo del lenguaje, la lectura y la escritura.

La ministra ha encargado un estudio sobre las cargas burocráticas que tienen los maestros
Diversos estudios subrayan que el formato papel es mejor que el digital en el aprendizaje de la lectura, según el profesor de Educación de la Universidad de Barcelona, Enric Prats. “Sabemos que lo que más ayuda es la lectura en papel, sosegada, larga y si se acompaña de la voz, mucho mejor”, apunta. Sin embargo, duda de que el cambio de digital al libro resuelva por sí mismo la comprensión lectora. A su juicio esto es un aspecto que compete a todos los maestros (no solo al de lengua) y que ayudaría generar un hábito de la lectura en los niños, aún sabiendo que es un objetivo que no se logrará en todos ellos.
Además de volver a los aprendizajes esenciales como la lectura, el Gobierno sueco también quiere que el maestro se centre en su labor docente. El Ministerio de Educación anunció el miércoles la creación de una comisión para analizar la carga burocrática de los profesores. “Las escuelas padecen la enfermedad de la documentación”, aseveró la ministra que aclaró que no se trata de que den más clases a cambio, sino de liberar tiempo para que los profesores puedan “planificar, revisar y llevar a cabo su labor docente, sin estar constantemente estresados por otras tareas”.



"""
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words= False, replace_whitespace = False
    )
print(wrapped_text)

