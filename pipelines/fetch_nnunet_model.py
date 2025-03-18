import os
import utilities.zenodo_link_nnunet as nnunet_zenodo
import requests

def nnunet_models(database):

    nnunet, nnunet_link= nnunet_zenodo.main()
    record_id = nnunet_link.split('.')[-1]

    nnunet_path = os.path.join(database.path(),nnunet)

    if os.path.exists(nnunet_path):
        database.log("nnunet was found in the local folder")
        return
    else:   

        zenodo_url = f"https://zenodo.org/records/{record_id}/files/{nnunet}?download=1"

        with requests.get(zenodo_url) as req:
                    with open(os.path.join(database.path(),nnunet), 'wb') as f:
                        for chunk in req.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)





