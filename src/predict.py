import sys
import json
import logging
from model import SpecialistMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def format_results(prediction):
    """Pretty print prediction results"""
    print("\n" + "="*65)
    print(f"🔍 SYMPTOM QUERY:")
    print(f"   {prediction['query']}")
    print(f"\n⏱  Inference Time: {prediction['inference_ms']}ms")
    print("="*65)
    print("🏥 TOP SPECIALIST RECOMMENDATIONS:")
    print("="*65)

    for i, result in enumerate(prediction["results"], 1):
        bar_length  = int(result["confidence"] / 2)
        bar         = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\n#{i} {result['specialist']}")
        print(f"    Confidence : {result['confidence']}%")
        print(f"    [{bar}]")
        print(f"    Matched    : {result['matched_symptoms'][:70]}...")
        print(f"    ℹ️  {result['description']}")

    print("\n" + "="*65)
    print("⚠️  DISCLAIMER: This is NOT a medical diagnosis.")
    print("    Please consult a licensed physician for proper evaluation.")
    print("="*65 + "\n")

def run_inference(symptom_text=None):
    """Run inference on a single query"""
    logger.info("Initializing SpecialistMapper for inference...")
    mapper = SpecialistMapper()

    if symptom_text is None:
        # Interactive mode
        print("\n" + "="*65)
        print("🏥  DOCTOR SYMPTOM TO SPECIALIST MAPPER")
        print("="*65)
        print("Describe your symptoms in plain language.")
        print("Type 'quit' to exit.\n")

        while True:
            symptom_text = input("📝 Your symptoms: ").strip()

            if symptom_text.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye! Stay healthy.")
                break

            if not symptom_text:
                print("⚠️  Please enter your symptoms.\n")
                continue

            try:
                prediction = mapper.predict(symptom_text)
                format_results(prediction)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
    else:
        # Single query mode
        prediction = mapper.predict(symptom_text)
        format_results(prediction)
        return prediction

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line: python src/predict.py "my symptoms here"
        query = " ".join(sys.argv[1:])
        run_inference(query)
    else:
        # Interactive mode
        run_inference()