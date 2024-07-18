import onnx_tool


def calculate_tops(onnx_path):
    onnx_tool.model_profile(onnx_path)
    # TOPS ~= 2 * MACs / 10^12


def main():
    # MACs: 2,391,273,060, TOPS: 0.00478254612
    # calculate_tops(r'Z:\8TSSD\AI_Tools\AiXin\model\PPVD_W640_H352.onnx')
    # MACs: 173,246,594, TOPS: 0.000346493188
    # calculate_tops(r'Z:\8TSSD\AI_Tools\AiXin\model\AI_cds_tempscale4.onnx')
    # MACs: 11,814,752, TOPS: 0.000023629504
    calculate_tops(r'Z:\8TSSD\AI_Tools\AiXin\model\2MPTZwhiteV5.onnx')


if __name__ == '__main__':
    main()
