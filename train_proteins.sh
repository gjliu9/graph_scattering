k=5

# python mGST_classification_proteins.py --variation $k --sample umin --gstcoe_norml MS
python mGST_classification_proteins.py --variation $k --sample umin --gstcoe_norml none
python mGST_classification_proteins.py --variation $k --sample umax --gstcoe_norml MS
python mGST_classification_proteins.py --variation $k --sample umax --gstcoe_norml none
