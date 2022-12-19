from frag_pele.Helpers import create_templates
from frag_pele.Covalent import correct_template_of_backbone_res
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', default=None, help='Input covalent-ligand PDB.')
parser.add_argument('template_name', help='Name of the template')
parser.add_argument('aminoacid_type', help='Name of the aminoacid type to which the covalent ligand is attached')

args=parser.parse_args()

input_pdb = args.input_pdb
template_name = args.template_name
aminoacid_type = args.aminoacid_type

create_templates.get_datalocal(input_pdb, template_name=template_name, aminoacid=True,
                               aminoacid_type=aminoacid_type)

correct_template_of_backbone_res.correct_template("DataLocal/Templates/OPLS2005/Protein/templates_generated/"+template_name.lower(),
                                                  "Data/Templates/OPLS2005/Protein/"+aminoacid_type.lower())
                                                  # the second argument is the relative path to the amino acid template in Data
