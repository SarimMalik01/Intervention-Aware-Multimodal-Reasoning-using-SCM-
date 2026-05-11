import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from configs.config import DEVICE
from models.scm_vlm import SCMVLM


def main() -> None:

    model = SCMVLM().to(DEVICE)
    model.eval()

    image_path = "images/Man_riding_bicycle.png"
    question = "What is the person holding?"

    # =========================
    # VISION ENCODER OUTPUT
    # =========================

    vision_output = model.vision(image_path)

    print("\n=========================")
    print("DETECTED OBJECTS")
    print("=========================\n")

    labels = vision_output["labels"][0]

    for i, label in enumerate(labels):
        print(f"Node {i}: {label}")

    # =========================
    # NORMAL FORWARD PASS
    # =========================

    output = model(image_path, question)

    print("\n=========================")
    print("MODEL OUTPUT (FACTUAL)")
    print("=========================\n")

    print("Logits:\n", output["logits"])

    adjacency = output["adjacency"]

    print("\nAdjacency Matrix Shape:\n", adjacency.shape)
    print("\nAdjacency Matrix:\n", adjacency)

    # =========================
    # COUNTERFACTUAL REASONING
    # =========================

    # print("\n=========================")
    # print("COUNTERFACTUAL ANALYSIS")
    # print("=========================\n")

    # objects = output["objects"]
    # adjacency = output["adjacency"]

    # # -------------------------
    # # STEP 1: Choose node
    # # -------------------------
    # # Example: assume node 1 = bicycle (you printed labels above)
    # node_index = torch.tensor([1]).to(DEVICE)

    # # -------------------------
    # # STEP 2: Create intervention
    # # -------------------------
    # # Option A: remove object (recommended for paper)
    # replacement = torch.zeros(
    #     1,
    #     objects.shape[-1],
    #     device=DEVICE
    # )

    # # -------------------------
    # # STEP 3: Run counterfactual
    # # -------------------------
    # cf_output = model.counterfactual(
    #     objects,
    #     adjacency,
    #     node_index,
    #     replacement
    # )

    # # -------------------------
    # # STEP 4: Compare results
    # # -------------------------
    # print("Factual logits:\n", output["logits"])
    # print("\nCounterfactual logits:\n", cf_output["logits"])

    # print("\nLogit shift:\n",
    #       cf_output["logits"] - output["logits"])


if __name__ == "__main__":
    main()