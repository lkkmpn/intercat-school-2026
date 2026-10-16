from ase.calculators.calculator import Calculator


def get_uma_calculator() -> Calculator:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit('uma-s-1p1', device='cuda')
    calc = FAIRChemCalculator(predictor, task_name='oc20')

    return calc
