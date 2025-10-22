#!/usr/bin/env python3
"""
Main script to run complete Legionella protein analysis pipeline

Usage:
    python run_analysis.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import ConfigManager, AnalysisLogger, DataManager, ensure_dir
from src.model_inference import run_full_analysis
from src.visualization import run_tsne_analysis, TSNEVisualizer
from src.dashboard import DashboardGenerator
from src.interpretability import compute_attributions_for_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Legionella protein analysis pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Skip model inference (use existing results)'
    )
    parser.add_argument(
        '--skip-tsne',
        action='store_true',
        help='Skip t-SNE computation (use existing results)'
    )
    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip dashboard generation'
    )
    return parser.parse_args()


def main():
    """Main pipeline execution"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print("="*70)
    print("LEGIONELLA PROTEIN ANALYSIS PIPELINE")
    print("="*70)
    print(f"\nLoading configuration from: {args.config}")

    config_manager = ConfigManager(config_path=args.config)

    # Save config if it doesn't exist
    if not Path(args.config).exists():
        print(f"Creating default configuration file: {args.config}")
        config_manager.save_config()

    config = config_manager.config

    # Setup paths
    model_path = config['model']['path']
    data_path = config['data']['input_csv']
    master_csv_path = config['data'].get('master_csv', None)  # Optional
    output_dir = config['data']['processed_dir']
    report_dir = config['output']['output_dir']

    # Ensure directories exist
    ensure_dir(output_dir)
    ensure_dir(report_dir)

    # Initialize logger
    logger = AnalysisLogger(
        log_dir=config['logging']['log_dir'],
        name='main_pipeline'
    )

    logger.info("Configuration loaded successfully")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    if master_csv_path:
        logger.info(f"Master CSV (true labels): {master_csv_path}")
    logger.info(f"Output directory: {output_dir}")

    start_time = time.time()

    # =========================================================================
    # STEP 1: Model Inference (Predictions + Embeddings + Attentions)
    # =========================================================================
    if not args.skip_inference:
        logger.info("\n" + "="*70)
        logger.info("STEP 1: RUNNING MODEL INFERENCE")
        logger.info("="*70)

        try:
            analysis_results = run_full_analysis(
                data_path=data_path,
                model_path=model_path,
                output_dir=output_dir,
                batch_size=config['model']['batch_size'],
                device=config['model']['device'],
                extract_embeddings=config['analysis']['extract_embeddings'],
                extract_attentions=config['analysis']['extract_attentions'],
                master_csv_path=master_csv_path,
                use_multi_gpu=config['model'].get('use_multi_gpu', True)
            )

            logger.info("Model inference completed successfully")

        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            raise

    else:
        logger.info("Skipping model inference (using existing results)")

    # =========================================================================
    # STEP 2: t-SNE Dimensionality Reduction
    # =========================================================================
    if config['analysis']['run_tsne'] and not args.skip_tsne:
        logger.info("\n" + "="*70)
        logger.info("STEP 2: RUNNING t-SNE ANALYSIS")
        logger.info("="*70)

        try:
            tsne_results = run_tsne_analysis(
                embeddings_path=f"{output_dir}/embeddings.npz",
                predictions_path=f"{output_dir}/predictions.csv",
                output_dir=output_dir,
                perplexity=config['analysis']['tsne_perplexity'],
                random_state=config['analysis']['random_seed'],
                highlight_proteins=config['visualization']['highlight_proteins']
            )

            logger.info("t-SNE analysis completed successfully")

            # Save t-SNE figures
            fig_2d = tsne_results['fig_2d']
            fig_3d = tsne_results['fig_3d']

            # Save interactive HTML versions
            fig_2d_path = Path(report_dir) / "tsne_2d_standalone.html"
            fig_3d_path = Path(report_dir) / "tsne_3d_standalone.html"

            fig_2d.write_html(str(fig_2d_path))
            fig_3d.write_html(str(fig_3d_path))

            logger.info(f"Standalone 2D plot saved: {fig_2d_path}")
            logger.info(f"Standalone 3D plot saved: {fig_3d_path}")

        except Exception as e:
            logger.error(f"t-SNE analysis failed: {str(e)}")
            raise

    else:
        logger.info("Skipping t-SNE analysis")
        # Load existing results
        try:
            data_manager = DataManager(base_dir=output_dir)
            visualizer = TSNEVisualizer(logger=logger)

            # Load data
            emb_data = data_manager.load_embeddings("embeddings.npz")
            predictions_df = data_manager.load_predictions("predictions.csv")
            tsne_data = data_manager.load_tsne_results("tsne_results.npz")

            # Recreate figures
            fig_2d = visualizer.create_tsne_plot_2d(
                tsne_coords=tsne_data['tsne_2d'],
                labels=tsne_data['labels'],
                metadata=predictions_df,
                color_by=config['visualization']['color_by'],
                highlight_proteins=config['visualization']['highlight_proteins'],
                title="2D t-SNE - Protein Embeddings"
            )

            fig_3d = visualizer.create_tsne_plot_3d(
                tsne_coords=tsne_data['tsne_3d'],
                labels=tsne_data['labels'],
                metadata=predictions_df,
                color_by=config['visualization']['color_by'],
                highlight_proteins=config['visualization']['highlight_proteins'],
                title="3D t-SNE - Protein Embeddings"
            )

        except Exception as e:
            logger.error(f"Could not load existing t-SNE results: {str(e)}")
            raise

    # =========================================================================
    # STEP 3: Compute Integrated Gradients Attributions (optional)
    # =========================================================================
    attribution_figs = None
    if config['analysis'].get('compute_attributions', False):
        logger.info("\n" + "="*70)
        logger.info("STEP 3: COMPUTING INTEGRATED GRADIENTS ATTRIBUTIONS")
        logger.info("="*70)

        try:
            # Load model and tokenizer
            logger.info(f"Loading model for attribution analysis...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # Load predictions
            data_manager = DataManager(base_dir=output_dir)
            predictions_df = data_manager.load_predictions("predictions.csv")

            # Determine proteins to analyze
            highlighted_proteins = config['visualization']['highlight_proteins']
            max_proteins = config['analysis'].get('attribution_max_proteins', 50)

            # Get top predicted effectors
            top_effectors = predictions_df.nlargest(max_proteins, 'probability_effector')
            proteins_to_analyze = set(top_effectors['label'].tolist())

            # Add highlighted proteins to the list
            if highlighted_proteins:
                proteins_to_analyze.update(highlighted_proteins)
                logger.info(f"Analyzing top {max_proteins} effectors + {len(highlighted_proteins)} highlighted proteins")
            else:
                logger.info(f"Analyzing top {len(proteins_to_analyze)} predicted effectors")

            proteins_to_analyze = list(proteins_to_analyze)

            # Compute attributions
            attribution_figs = compute_attributions_for_dataset(
                model=model,
                tokenizer=tokenizer,
                predictions_df=predictions_df,
                output_dir=output_dir,
                protein_ids=proteins_to_analyze,
                device=config['model']['device'],
                use_multi_gpu=config['model'].get('use_multi_gpu', True),
                model_path=model_path
            )

            logger.info(f"Computed attributions for {len(attribution_figs)} proteins")

        except Exception as e:
            logger.error(f"Attribution analysis failed: {str(e)}")
            logger.warning("Continuing without attributions...")
            attribution_figs = None

    # =========================================================================
    # STEP 4: Generate Dashboard
    # =========================================================================
    if not args.skip_dashboard:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: GENERATING DASHBOARD")
        logger.info("="*70)

        try:
            dashboard = DashboardGenerator(logger=logger)
            data_manager = DataManager(base_dir=output_dir)

            # Load all data
            predictions_df = data_manager.load_predictions("predictions.csv")

            # Load attentions if available
            attentions_data = None
            if config['analysis']['extract_attentions']:
                try:
                    attentions_data = data_manager.load_attentions("attentions.npz")

                    # Also load sequences for attention visualization
                    emb_data = data_manager.load_embeddings("embeddings.npz")
                    attentions_data['sequences'] = emb_data['sequences']

                    # Load attention masks
                    import numpy as np
                    masks_data = np.load(f"{output_dir}/attention_masks.npz")
                    attentions_data['attention_masks'] = masks_data['attention_masks']

                except Exception as e:
                    logger.warning(f"Could not load attentions: {e}")

            # Generate report
            report_path = dashboard.generate_html_report(
                predictions_df=predictions_df,
                fig_tsne_2d=fig_2d,
                fig_tsne_3d=fig_3d,
                attentions_data=attentions_data,
                highlight_proteins=config['visualization']['highlight_proteins'],
                output_path=f"{report_dir}/{config['output']['report_name']}",
                attribution_figs=attribution_figs
            )

            logger.info(f"Dashboard generated: {report_path}")

        except Exception as e:
            logger.error(f"Dashboard generation failed: {str(e)}")
            raise

    else:
        logger.info("Skipping dashboard generation")

    # =========================================================================
    # COMPLETION
    # =========================================================================
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    logger.info(f"\nResults saved in:")
    logger.info(f"  - Processed data: {output_dir}/")
    logger.info(f"  - Reports: {report_dir}/")
    logger.info(f"  - Main report: {report_dir}/{config['output']['report_name']}")
    logger.info("\n" + "="*70)

    print("\n" + "="*70)
    print("SUCCESS! Analysis pipeline completed.")
    print("="*70)
    print(f"\nOpen the report in your browser:")
    print(f"  {Path(report_dir).absolute()}/{config['output']['report_name']}")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Pipeline failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
