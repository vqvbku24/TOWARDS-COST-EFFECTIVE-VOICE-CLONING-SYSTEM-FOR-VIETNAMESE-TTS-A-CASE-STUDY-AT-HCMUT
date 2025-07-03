import argparse
import os
import numpy as np
import soundfile
from inference import infer_tool
from inference.infer_tool import Svc

# Disable numba warnings
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

infer_tool.mkdir(["raw", "results"])
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")


def main():
    parser = argparse.ArgumentParser(description='Inference with optional speaker embedding (.npy)')

    # Required
    parser.add_argument('-n', '--input_wav', type=str, required=True, help='Path to input wav file')
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth", help='Path to model checkpoint')
    parser.add_argument('-c', '--config_path', type=str, default="configs/config.json", help='Path to config.json')

    # Optional mode
    parser.add_argument('--use_embedding', action='store_true', help='Use external speaker embedding (.npy)')
    parser.add_argument('--embedding_path', type=str, default=None, help='Path to speaker embedding .npy file')
    
    parser.add_argument('-s', '--speaker', type=str, default="speaker_1", help='Target speaker name (if using cluster-based mode)')
    parser.add_argument('--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='Cluster model path')
    parser.add_argument('--cluster_ratio', type=float, default=0.75, help='Cluster infer ratio')

    # Audio settings
    parser.add_argument('--f0_predictor', type=str, default="harvest", help='F0 predictor: pm, harvest, dio, etc.')
    parser.add_argument('--auto_predict_f0', action='store_true', help='Use auto F0 prediction')
    parser.add_argument('--trans', type=int, default=0, help='Pitch shift (semitones)')
    parser.add_argument('--slice_db', type=int, default=-40, help='Silence threshold')
    parser.add_argument('--output_format', type=str, default='flac', help='Output format: wav, flac, etc.')

    args = parser.parse_args()

    # Build model
    svc_model = Svc(
        model_path=args.model_path,
        config_path=args.config_path,
        device=None,
        cluster_model_path=args.cluster_model_path,
        enhancer=False,
        diffusion_model_path=None,
        diffusion_config_path=None,
        shallow_diffusion=False,
        only_diffusion=False,
        use_spk_mix=False,
        feature_retrieval=False
    )

    # Format and prepare input
    input_wav = args.input_wav
    infer_tool.format_wav(input_wav)
    raw_audio_path = f"raw/{os.path.basename(input_wav)}"
    os.rename(input_wav, raw_audio_path)  # Move to raw/

    kwargs = {
        "raw_audio_path": raw_audio_path,
        "tran": args.trans,
        "slice_db": args.slice_db,
        "cluster_infer_ratio": args.cluster_ratio,
        "auto_predict_f0": args.auto_predict_f0,
        "f0_predictor": args.f0_predictor,
        "clip_seconds": 0,
        "lg_num": 0,
        "lgr_num": 0.75,
        "noice_scale": 0.4,
        "pad_seconds": 0.5,
        "enhancer_adaptive_key": 0,
        "cr_threshold": 0.05,
        "k_step": 100,
        "use_spk_mix": False,
        "second_encoding": False,
        "loudness_envelope_adjustment": 1.0,
    }

    # Mode 1: Using speaker embedding
    if args.use_embedding:
        if not args.embedding_path or not os.path.exists(args.embedding_path):
            raise FileNotFoundError("Please provide a valid --embedding_path when --use_embedding is set.")
        speaker_embedding = np.load(args.embedding_path)
        kwargs["speaker_embedding"] = speaker_embedding
        spk_name = "embed"
    else:
        kwargs["spk"] = args.speaker
        spk_name = args.speaker

    # Run inference
    print("🔊 Running inference...")
    audio = svc_model.slice_inference(**kwargs)

    # Save result
    output_name = f"{os.path.basename(raw_audio_path).replace('.wav', '')}_{spk_name}.{args.output_format}"
    output_path = os.path.join("results", output_name)
    soundfile.write(output_path, audio, svc_model.target_sample, format=args.output_format)

    print(f"Done! Saved to {output_path}")


if __name__ == '__main__':
    main()
