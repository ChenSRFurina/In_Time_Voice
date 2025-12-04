import sounddevice as sd
import sys

# ----------------------------------------------------
# 查找所有设备并定位 CABLE-B Input 的脚本
# ----------------------------------------------------

try:
    print("===============================================================")
    print("正在查找所有音频设备索引 (包括输入和输出):")
    devices = sd.query_devices()
    cable_b_input_index = None

    for i, device in enumerate(devices):
        # max_output_channels > 0 表示这是一个播放/输出设备
        max_output = device.get('max_output_channels', 0)
        
        # 打印设备信息
        output_info = f" | 输出通道: {max_output}"
        
        # 检查是否是 CABLE-B Input
        if max_output > 0 and "CABLE-B Input" in device['name']:
            cable_b_input_index = i
            print(f"✅ 索引 {i}: {device['name']}{output_info} <--- 目标播放设备 (Y)")
        else:
            print(f"索引 {i}: {device['name']}{output_info}")

    if cable_b_input_index is None:
        print("\n⚠️ 警告：未在播放设备中找到名称包含 'CABLE-B Input' 的设备。请确认 VB-CABLE B 已安装。")
    else:
        print(f"\n✅ 目标 CABLE-B Input 索引 (Y) 查找完成: {cable_b_input_index}")
        print("===============================================================")
        print("现在你可以使用这个索引运行扬声器翻译实例了。")

except Exception as e:
    # 捕获可能出现的 sounddevice 错误，例如 UnboundLocalError
    print(f"❌ 运行脚本时发生错误: {e}", file=sys.stderr)
    print("请确保 sounddevice 和 numpy 已安装，并且你的环境中没有 UnboundLocalError 问题。", file=sys.stderr)