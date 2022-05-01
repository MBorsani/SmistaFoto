import face_recognition as fr
import os
import shutil

from face_recognition.api import face_locations
#from progress.bar import ChargingBar

cwd = os.path.dirname(os.path.realpath(__file__))
personeDIR = cwd + "/Persone"
fotoDIR = cwd + "/Tutte le foto"
risDIR = cwd + "/Foto divise"

for persona in os.scandir(personeDIR):
    print(persona.name)
    personaImage = fr.load_image_file(personeDIR + "/" + persona.name)
    personaLocations = fr.face_locations(personaImage, number_of_times_to_upsample=0, model="cnn")
    personaEncondings = fr.face_encodings(personaImage, personaLocations)
    
    if len(personaEncondings) > 0:
            personaEnconding = personaEncondings[0]
    else:
            print(persona.name + ": nessuna faccia trovata")
            continue

    diviseDIR = risDIR + "/" + persona.name.split(".")[0]
    os.mkdir(diviseDIR)
    
    for foto in os.scandir(fotoDIR):
        fotoImage = fr.load_image_file(fotoDIR + "/" + foto.name)
        fotoLocations = fr.face_locations(fotoImage, number_of_times_to_upsample=0, model="cnn")
        fotoEncondings = fr.face_encodings(fotoImage, fotoLocations)
        if len(fotoEncondings) > 0:
            for encoding in fotoEncondings:
                res = fr.compare_faces([personaEnconding], encoding)

                if True in res:
                    shutil.copy2(fotoDIR + "/" + foto.name, diviseDIR)
        else:
            print(foto.name + ": nessuna faccia trovata")

        


print("Fatto!")
 
