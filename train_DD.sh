k=5
python pGST_classification_DD.py --variation $k --sample umin --gstcoe_norml MS
python pGST_classification_DD.py --variation $k --sample umin --gstcoe_norml none
python pGST_classification_DD.py --variation $k --sample umin --gstcoe_norml MS --fake_signal degrees
python pGST_classification_DD.py --variation $k --sample umin --gstcoe_norml none --fake_signal degrees
python pGST_classification_DD.py --variation $k --sample umax --gstcoe_norml MS
python pGST_classification_DD.py --variation $k --sample umax --gstcoe_norml none
python pGST_classification_DD.py --variation $k --sample umax --gstcoe_norml MS --fake_signal degrees
python pGST_classification_DD.py --variation $k --sample umax --gstcoe_norml none --fake_signal degrees