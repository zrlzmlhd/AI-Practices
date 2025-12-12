"""
Transformer机器翻译评估脚本

使用方法:
    python src/evaluate.py --model_path models/transformer_translation_model.h5 --processor_path models/translation_processor.pkl
    python src/evaluate.py --test_src data/test.en --test_tgt data/test.zh

【评估指标】:
- BLEU分数（机器翻译标准指标）
- 词级准确率
- 翻译样本展示
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import TranslationDataProcessor
from src.model import TransformerTranslationModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估Transformer机器翻译模型')

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--processor_path', type=str, required=True,
                       help='处理器文件路径')
    parser.add_argument('--test_src', type=str, default='data/test.en',
                       help='测试集源语言文件')
    parser.add_argument('--test_tgt', type=str, default='data/test.zh',
                       help='测试集目标语言文件')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大测试样本数')
    parser.add_argument('--num_display', type=int, default=10,
                       help='展示的翻译样本数')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果保存目录')

    return parser.parse_args()


def calculate_bleu_score(references, hypotheses, max_n=4):
    """
    计算BLEU分数

    【是什么】：机器翻译质量评估的标准指标
    【如何计算】：
        - 计算n-gram精确率（n=1,2,3,4）
        - 应用简短惩罚（BP）
        - BLEU = BP * exp(sum(log(p_n)))

    Args:
        references: 参考翻译列表（每个是词列表）
        hypotheses: 模型翻译列表（每个是词列表）
        max_n: 最大n-gram长度

    Returns:
        BLEU分数字典
    """
    from collections import defaultdict

    def get_ngrams(tokens, n):
        """获取n-gram"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    def count_ngrams(tokens, n):
        """统计n-gram频率"""
        ngrams = get_ngrams(tokens, n)
        return Counter(ngrams)

    # 计算每个n的精确率
    precisions = []

    for n in range(1, max_n + 1):
        matched = 0
        total = 0

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = count_ngrams(ref, n)
            hyp_ngrams = count_ngrams(hyp, n)

            # 计算匹配的n-gram数量
            for ngram, count in hyp_ngrams.items():
                matched += min(count, ref_ngrams.get(ngram, 0))

            total += max(len(hyp) - n + 1, 0)

        if total > 0:
            precisions.append(matched / total)
        else:
            precisions.append(0.0)

    # 计算简短惩罚（Brevity Penalty）
    ref_len = sum(len(ref) for ref in references)
    hyp_len = sum(len(hyp) for hyp in hypotheses)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    # 计算BLEU分数
    if min(precisions) > 0:
        log_precisions = [np.log(p) for p in precisions]
        bleu = bp * np.exp(sum(log_precisions) / len(log_precisions))
    else:
        bleu = 0.0

    return {
        'BLEU': bleu,
        'BLEU-1': precisions[0] if len(precisions) > 0 else 0.0,
        'BLEU-2': precisions[1] if len(precisions) > 1 else 0.0,
        'BLEU-3': precisions[2] if len(precisions) > 2 else 0.0,
        'BLEU-4': precisions[3] if len(precisions) > 3 else 0.0,
        'BP': bp,
        'ref_len': ref_len,
        'hyp_len': hyp_len
    }


def translate_and_evaluate(model, processor, src_sequences, tgt_sequences):
    """
    翻译并评估

    【流程】：
    1. 对每个源句子进行翻译
    2. 与参考翻译对比
    3. 计算BLEU分数
    """
    print("\n" + "="*60)
    print("翻译测试集")
    print("="*60)

    references = []
    hypotheses = []

    for i, (src_seq, tgt_seq) in enumerate(zip(src_sequences, tgt_sequences)):
        if (i + 1) % 100 == 0:
            print(f"  已翻译: {i+1}/{len(src_sequences)}")

        # 参考翻译（去除特殊token）
        ref_words = []
        for idx in tgt_seq:
            if idx == 0:  # PAD
                break
            word = processor.tgt_idx2word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                ref_words.append(word)

        # 模型翻译
        translation = model.translate(
            src_seq,
            processor.tgt_word2idx,
            processor.tgt_idx2word,
            max_len=processor.max_len
        )
        hyp_words = translation.split()

        references.append(ref_words)
        hypotheses.append(hyp_words)

    print(f"  完成翻译: {len(src_sequences)}个句子")

    # 计算BLEU分数
    bleu_scores = calculate_bleu_score(references, hypotheses)

    return references, hypotheses, bleu_scores


def display_translation_samples(processor, src_sequences, references, hypotheses, num_samples=10):
    """
    展示翻译样本

    【是什么】：对比源句子、参考翻译和模型翻译
    【为什么】：直观评估翻译质量
    """
    print("\n" + "="*60)
    print("翻译样本展示")
    print("="*60)

    for i in range(min(num_samples, len(src_sequences))):
        # 源句子
        src_seq = src_sequences[i]
        src_words = []
        for idx in src_seq:
            if idx == 0:
                break
            word = processor.src_idx2word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<UNK>']:
                src_words.append(word)

        print(f"\n样本 {i+1}:")
        print(f"  源句子: {' '.join(src_words)}")
        print(f"  参考翻译: {' '.join(references[i])}")
        print(f"  模型翻译: {' '.join(hypotheses[i])}")


def plot_bleu_scores(bleu_scores, save_path):
    """
    绘制BLEU分数

    【可视化】：展示不同n-gram的BLEU分数
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'BLEU']
    values = [bleu_scores[m] for m in metrics]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('BLEU分数评估', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ BLEU分数图已保存: {save_path}")
    plt.close()


def analyze_translation_length(references, hypotheses, save_path):
    """
    分析翻译长度分布

    【是什么】：对比参考翻译和模型翻译的长度
    【为什么】：检查模型是否倾向于生成过长或过短的翻译
    """
    ref_lengths = [len(ref) for ref in references]
    hyp_lengths = [len(hyp) for hyp in hypotheses]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 长度分布直方图
    axes[0].hist(ref_lengths, bins=30, alpha=0.7, label='参考翻译', color='blue', edgecolor='black')
    axes[0].hist(hyp_lengths, bins=30, alpha=0.7, label='模型翻译', color='red', edgecolor='black')
    axes[0].set_xlabel('句子长度（词数）', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('翻译长度分布', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 长度对比散点图
    axes[1].scatter(ref_lengths, hyp_lengths, alpha=0.5, s=20)
    max_len = max(max(ref_lengths), max(hyp_lengths))
    axes[1].plot([0, max_len], [0, max_len], 'r--', linewidth=2, label='理想线')
    axes[1].set_xlabel('参考翻译长度', fontsize=12)
    axes[1].set_ylabel('模型翻译长度', fontsize=12)
    axes[1].set_title('翻译长度对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 长度分析图已保存: {save_path}")
    plt.close()


def main():
    """主评估流程"""
    args = parse_args()

    print("="*60)
    print("Transformer机器翻译 - 模型评估")
    print("="*60)

    # 创建结果目录
    project_dir = Path(__file__).parent.parent
    result_dir = project_dir / args.result_dir
    result_dir.mkdir(exist_ok=True)

    # ============================================
    # 1. 加载模型和处理器
    # ============================================
    print("\n" + "="*60)
    print("步骤1: 加载模型和处理器")
    print("="*60)

    # 加载处理器
    processor = TranslationDataProcessor()
    processor.load_processor(args.processor_path)
    print(f"✓ 处理器已加载")
    print(f"  源语言词汇表: {len(processor.src_word2idx)}")
    print(f"  目标语言词汇表: {len(processor.tgt_word2idx)}")

    # 加载模型
    model = keras.models.load_model(args.model_path)
    print(f"✓ 模型已加载")

    # 重新包装为TransformerTranslationModel
    translator = TransformerTranslationModel(
        src_vocab_size=len(processor.src_word2idx),
        tgt_vocab_size=len(processor.tgt_word2idx),
        max_len=processor.max_len
    )
    translator.model = model

    # ============================================
    # 2. 加载测试数据
    # ============================================
    print("\n" + "="*60)
    print("步骤2: 加载测试数据")
    print("="*60)

    try:
        src_sentences, tgt_sentences = processor.load_parallel_data(
            args.test_src,
            args.test_tgt,
            max_samples=args.max_samples
        )

        if not src_sentences:
            raise FileNotFoundError("无法加载测试数据")

        # 编码和填充
        src_encoded = processor.encode_sentences(src_sentences, 'src')
        tgt_encoded = processor.encode_sentences(tgt_sentences, 'tgt', add_sos_eos=True)

        src_test = processor.pad_sequences(src_encoded)
        tgt_test = processor.pad_sequences(tgt_encoded)

        print(f"✓ 测试数据已加载")
        print(f"  测试样本数: {len(src_test)}")

    except FileNotFoundError as e:
        print(f"\n✗ 测试数据文件不存在: {e}")
        print("请确保测试数据文件存在")
        return

    # ============================================
    # 3. 翻译和评估
    # ============================================
    print("\n" + "="*60)
    print("步骤3: 翻译和评估")
    print("="*60)

    references, hypotheses, bleu_scores = translate_and_evaluate(
        translator, processor, src_test, tgt_test
    )

    # 打印BLEU分数
    print("\n" + "="*60)
    print("BLEU分数")
    print("="*60)
    print(f"  BLEU: {bleu_scores['BLEU']:.4f}")
    print(f"  BLEU-1: {bleu_scores['BLEU-1']:.4f}")
    print(f"  BLEU-2: {bleu_scores['BLEU-2']:.4f}")
    print(f"  BLEU-3: {bleu_scores['BLEU-3']:.4f}")
    print(f"  BLEU-4: {bleu_scores['BLEU-4']:.4f}")
    print(f"  简短惩罚(BP): {bleu_scores['BP']:.4f}")
    print(f"  参考长度: {bleu_scores['ref_len']}")
    print(f"  翻译长度: {bleu_scores['hyp_len']}")

    # ============================================
    # 4. 展示翻译样本
    # ============================================
    display_translation_samples(
        processor, src_test, references, hypotheses,
        num_samples=args.num_display
    )

    # ============================================
    # 5. 保存结果
    # ============================================
    print("\n" + "="*60)
    print("步骤4: 保存结果")
    print("="*60)

    # 绘制BLEU分数
    bleu_plot_path = result_dir / 'bleu_scores.png'
    plot_bleu_scores(bleu_scores, bleu_plot_path)

    # 分析翻译长度
    length_plot_path = result_dir / 'translation_length_analysis.png'
    analyze_translation_length(references, hypotheses, length_plot_path)

    # 保存评估结果
    eval_results_path = result_dir / 'evaluation_results.txt'
    with open(eval_results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Transformer机器翻译 - 评估结果\n")
        f.write("="*60 + "\n\n")

        f.write("BLEU分数:\n")
        f.write(f"  BLEU: {bleu_scores['BLEU']:.4f}\n")
        f.write(f"  BLEU-1: {bleu_scores['BLEU-1']:.4f}\n")
        f.write(f"  BLEU-2: {bleu_scores['BLEU-2']:.4f}\n")
        f.write(f"  BLEU-3: {bleu_scores['BLEU-3']:.4f}\n")
        f.write(f"  BLEU-4: {bleu_scores['BLEU-4']:.4f}\n")
        f.write(f"  简短惩罚(BP): {bleu_scores['BP']:.4f}\n\n")

        f.write("翻译样本:\n")
        f.write("="*60 + "\n")
        for i in range(min(args.num_display, len(src_test))):
            src_words = []
            for idx in src_test[i]:
                if idx == 0:
                    break
                word = processor.src_idx2word.get(idx, '<UNK>')
                if word not in ['<PAD>', '<UNK>']:
                    src_words.append(word)

            f.write(f"\n样本 {i+1}:\n")
            f.write(f"  源句子: {' '.join(src_words)}\n")
            f.write(f"  参考翻译: {' '.join(references[i])}\n")
            f.write(f"  模型翻译: {' '.join(hypotheses[i])}\n")

    print(f"✓ 评估结果已保存: {eval_results_path}")

    # ============================================
    # 总结
    # ============================================
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    print(f"✓ BLEU分数: {bleu_scores['BLEU']:.4f}")
    print(f"✓ 评估结果已保存: {eval_results_path}")
    print(f"✓ BLEU分数图已保存: {bleu_plot_path}")
    print(f"✓ 长度分析图已保存: {length_plot_path}")


if __name__ == '__main__':
    main()
