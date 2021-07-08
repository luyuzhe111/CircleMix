import xmltodict
import json
import os

loop = 1

if loop == 1:
    result_dir = 'pipeline/thd_0.01_result'
else:
    result_dir = 'pipeline/thd_0.01_result_2'

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

    glom = []
    for idx, patch in enumerate(det_patches):
        if float(patch['@Text']) > 0.5 or patch['@Id'] not in non_glom_id:
            glom_patch = patch.copy()
            glom_patch['@Id'] = len(glom) + 1
            glom.append(glom_patch)

    doc['Annotations']['Annotation']['Regions']['Region'] = glom

    out = xmltodict.unparse(doc, pretty=True)
    xml_file = f'{root_dir}/ftd_patch.xml'
    with open(xml_file, 'wb') as file:
        file.write(out.encode('utf-8'))
