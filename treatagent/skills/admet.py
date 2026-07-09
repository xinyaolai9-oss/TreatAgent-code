try:
    from admet_ai import ADMETModel
except Exception:  # pragma: no cover - optional runtime dependency
    ADMETModel = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional runtime dependency
    Chem = None


def fix_smiles_errors(smiles):
    open_brackets = smiles.count("[")
    close_brackets = smiles.count("]")
    if open_brackets > close_brackets:
        smiles += "]" * (open_brackets - close_brackets)
    return smiles


def admet_data(smiles):
    if not smiles or not isinstance(smiles, str):
        return None
    if ADMETModel is None or Chem is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fixed_smiles = fix_smiles_errors(smiles)
            mol = Chem.MolFromSmiles(fixed_smiles)
            if mol is None:
                return None
            smiles = fixed_smiles
    except Exception:
        return None

    try:
        model = ADMETModel()
        admet_prediction_data = model.predict(smiles=smiles)

        absorption = (
            "Absorption\n"
            "Bioavailability: Bioavailability measures the fraction of an administered drug dose that reaches systemic circulation in an active form.\n"
            "Human Intestinal Absorption (HIA): HIA assesses how well a drug is absorbed through the intestinal tract, essential for orally administered drugs.\n"
            "Lipophilicity: Lipophilicity, often measured by the logP value (the partition coefficient between octanol and water), reflects a drug's ability to dissolve in fats versus water.\n"
            "Aqueous Solubility: Aqueous solubility measures how well a drug dissolves in water, crucial for absorption and bioavailability.\n"
        )

        distribution = (
            "Distribution\n"
            "Blood-Brain Barrier Penetration: Blood-Brain Barrier(BBB) penetration measures a drug's ability to cross the blood-brain barrier, which is critical for drugs targeting central nervous system (CNS) disorders like neurological diseases or psychiatric conditions.\n"
            "Plasma Protein Binding Rate (PPBR): Plasma protein binding rate represents the percentage of a drug bound to plasma proteins in the blood.\n"
            "Volume of Distribution at Steady State (Vdss): Volume of Distribution at Steady State (Vdss) is a pharmacokinetic measure that indicates the extent of drug distribution throughout the body relative to the plasma.\n"
        )

        metabolism = (
            "Metabolism\n"
            "CYP Inhibition: CYP inhibition refers to a drug's ability to inhibit cytochrome.\n"
            "CYP Substrate: CYP substrate potential indicates whether a drug itself is metabolized by specific CYP enzymes.\n"
        )

        excretion = (
            "Excretion\n"
            "Half-Life: Half-life is the time required for the concentration of a drug in the bloodstream to decrease by half.\n"
            "Drug Clearance (Hepatocyte): Hepatocyte clearance refers to the rate at which liver cells metabolize and eliminate a drug.\n"
            "Drug Clearance (Microsome): Microsome clearance assesses the rate of metabolism by liver microsomes.\n"
        )

        toxicity = (
            "Toxicity\n"
            "hERG Blocking: hERG (human Ether-a-go-go-Related Gene) refers to a drug's potential to inhibit the hERG potassium ion channel, which is critical for heart function.\n"
            "Clinical Toxicity: Clinical toxicity assesses the likelihood of a drug causing adverse effects in humans based on both preclinical and clinical data.\n"
            "Drug-Induced Liver Injury (DILI): Drug-induced liver injury evaluates the potential risk of liver damage from a drug.\n"
            "AMES: The AMES test predicts mutagenic potential, identifying the likelihood that a drug could cause genetic mutations, which is important for assessing potential carcinogenicity.\n"
        )

        return (
            f"{absorption}"
            f"Bioavailability: {admet_prediction_data['Bioavailability_Ma']:.2f}\n"
            f"HIA: {admet_prediction_data['HIA_Hou']:.3f}\n"
            f"Lipophilicity (logP): {admet_prediction_data['logP']:.3f}\n"
            f"Aqueous Solubility: {admet_prediction_data['Solubility_AqSolDB']:.3f}\n\n"
            f"{distribution}"
            f"BBB Penetration: {admet_prediction_data['BBB_Martins']:.3f}\n"
            f"PPBR: {admet_prediction_data['PPBR_AZ']:.3f}\n"
            f"Vdss: {admet_prediction_data['VDss_Lombardo']:.3f}\n\n"
            f"{metabolism}"
            f"CYP Inhibition: {admet_prediction_data['CYP3A4_Veith']:.3f}\n"
            f"CYP Substrate: {admet_prediction_data['CYP3A4_Substrate_CarbonMangels']:.3f}\n\n"
            f"{excretion}"
            f"Half-Life: {admet_prediction_data['Half_Life_Obach']:.3f}\n"
            f"Clearance (Hepatocyte): {admet_prediction_data['Clearance_Hepatocyte_AZ']:.3f}\n"
            f"Clearance (Microsome): {admet_prediction_data['Clearance_Microsome_AZ']:.3f}\n\n"
            f"{toxicity}"
            f"hERG Blocking: {admet_prediction_data['hERG']:.3f}\n"
            f"Clinical Toxicity: {admet_prediction_data['ClinTox']:.3f}\n"
            f"DILI: {admet_prediction_data['DILI']:.3f}\n"
            f"AMES: {admet_prediction_data['AMES']:.3f}\n"
        )
    except Exception:
        return None
