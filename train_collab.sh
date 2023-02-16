k=5
python mGST_classification_COLLAB.py --variation $k --sample umin --gstcoe_norml MS --fake_signal degrees
python mGST_classification_COLLAB.py --variation $k --sample umin --gstcoe_norml none --fake_signal degrees
python mGST_classification_COLLAB.py --variation $k --sample umin --gstcoe_norml MS
python mGST_classification_COLLAB.py --variation $k --sample umin --gstcoe_norml none
python mGST_classification_COLLAB.py --variation $k --sample umax --gstcoe_norml MS
python mGST_classification_COLLAB.py --variation $k --sample umax --gstcoe_norml none
python mGST_classification_COLLAB.py --variation $k --sample umax --gstcoe_norml MS --fake_signal degrees
python mGST_classification_COLLAB.py --variation $k --sample umax --gstcoe_norml none --fake_signal degrees
