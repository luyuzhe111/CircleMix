import xmltodict
import json
import os
import sys

loop = int(sys.argv[1])
threshold = float(sys.argv[2])
result_dir = f'pipeline/thd_{threshold}_result_{loop}' if loop == 2 else f'pipeline/thd_{threshold}_result'

for case in os.listdir(result_dir):
    print(case)
    root_dir = f'{result_dir}/{case}'
    xml_file = f'{root_dir}/{case}.xml'
    pred_file = f'{root_dir}/patch_pred.json'

    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())

    with open(pred_file) as f:
        pred = json.load(f)

    det_patches = doc['Annotations']['Annotation']['Regions']['Region']

    non_glom = [i for i in pred if i['pred'] == 1]
    non_glom_idx = [int(os.path.basename(i['image_dir']).split('-x-')[1].split('_')[1]) for i in non_glom]
    non_glom_id = [str(i + 1) for i in non_glom_idx]

    print(f'{len(det_patches)} detected glomeruli, {len(det_patches)  -  len(non_glom_id)} filtered glomeruli')

    glom = []
    for idx, patch in enumerate(det_patches):
        if patch['@Id'] not in non_glom_id:
            glom_patch = patch.copy()
            glom_patch['@Id'] = len(glom) + 1
            glom.append(glom_patch)

    doc['Annotations']['Annotation']['Regions']['Region'] = glom

    out = xmltodict.unparse(doc, pretty=True)
    xml_file = f'{root_dir}/ftd_patch.xml'
    with open(xml_file, 'wb') as file:
        file.write(out.encode('utf-8'))
