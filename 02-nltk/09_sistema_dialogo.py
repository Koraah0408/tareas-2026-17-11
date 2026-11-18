# Diapositivas 1, 9 y 54: Applications (nltk.chat), Arquitectura para sistemas de dialogo
# y el esquema de tutor inteligente (modelo del estudiante, dominio del conocimiento,
# estrategias de ensenanza, interfaz de usuario).
#
# Es un chatbot sencillo hecho con nltk.chat.util.Chat (como los chatbots de nltk.chat)
# que hace de "tutor" de NLTK. Escribe 'salir' para terminar.
#   python 09_sistema_dialogo.py           -> modo interactivo
#   python 09_sistema_dialogo.py --demo    -> corre una conversacion de ejemplo
import sys
from nltk.chat.util import Chat, reflections

# Dominio del conocimiento: patron (regex) -> posibles respuestas
pares = [
    (r"(hola|buenas)(.*)",
     ["Hola! Soy tu tutor de NLTK. Preguntame por: tokenizar, etiquetar, wordnet, chunking o clasificar."]),
    (r"(.*)tokeni(.*)",
     ["Para tokenizar usa nltk.word_tokenize('texto') y para oraciones nltk.sent_tokenize('texto')."]),
    (r"(.*)etiquet(.*)|(.*)pos(.*)",
     ["Para etiquetar partes de la oracion: nltk.pos_tag(nltk.word_tokenize('texto'))."]),
    (r"(.*)wordnet(.*)|(.*)sinonim(.*)",
     ["Con WordNet: from nltk.corpus import wordnet as wn; wn.synsets('car')."]),
    (r"(.*)chunk(.*)",
     ["Un chunker se hace con nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}')."]),
    (r"(.*)clasific(.*)",
     ["Para clasificar: nltk.NaiveBayesClassifier.train(train_set) y luego classifier.classify(rasgos)."]),
    (r"me llamo (.*)",
     ["Mucho gusto, %1. Que quieres aprender de NLTK?"]),
    (r"(.*)gracias(.*)",
     ["De nada! Algo mas?"]),
    (r"salir",
     ["Hasta luego!"]),
    (r"(.*)",
     ["No entendi. Prueba con: tokenizar, etiquetar, wordnet, chunking o clasificar.",
      "Puedes reformularlo? Se de tokenizar, etiquetar, wordnet, chunking y clasificar."]),
]

tutor = Chat(pares, reflections)

if "--demo" in sys.argv:
    for frase in ["Hola", "me llamo Ana", "como tokenizo un texto?", "y wordnet?",
                  "como hago un chunk?", "gracias", "salir"]:
        print("Tu   :", frase)
        print("Tutor:", tutor.respond(frase.lower()))
else:
    print("Tutor de NLTK (escribe 'salir' para terminar)")
    tutor.converse(quit="salir")
