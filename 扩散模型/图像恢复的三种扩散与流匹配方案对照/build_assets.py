"""Rebuild the original schematic figures and resolve existing report links."""
from pathlib import Path
from urllib.parse import quote
import math
import os
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IMAGES = HERE / 'images'
IMAGES.mkdir(exist_ok=True)
FONT_PATH = Path('C:/Windows/Fonts/msyh.ttc')
FONTS = {n: ImageFont.truetype(str(FONT_PATH), n) for n in (22, 25, 28, 32, 38)}
INK = '#243247'
BLUE = '#2867aa'
GREEN = '#19765d'
ORANGE = '#ac5b19'

def canvas(title, subtitle, height=760):
    im = Image.new('RGB', (1680, height), '#f7f9fc')
    d = ImageDraw.Draw(im)
    d.text((55, 30), title, font=FONTS[38], fill=INK)
    d.text((55, 90), subtitle, font=FONTS[22], fill='#536277')
    return im, d

def box(d, rect, lines, color=BLUE):
    x1,y1,x2,y2 = rect
    d.rounded_rectangle(rect, radius=16, fill='white', outline=color, width=3)
    lines = lines.split('\n')
    h = 38 * len(lines)
    for i,line in enumerate(lines):
        bounds = d.textbbox((0,0), line, font=FONTS[25])
        width = bounds[2]-bounds[0]
        d.text(((x1+x2-width)/2, (y1+y2-h)/2+i*38), line, font=FONTS[25], fill=INK)

def arrow(d, p, q, color=BLUE, dashed=False):
    dx,dy = q[0]-p[0], q[1]-p[1]
    length=math.hypot(dx,dy)
    if dashed:
        for a in range(0,int(length)-12,19):
            b=min(a+10,length-12)
            d.line((p[0]+dx*a/length,p[1]+dy*a/length,p[0]+dx*b/length,p[1]+dy*b/length),fill=color,width=3)
    else:
        d.line((p,q),fill=color,width=4)
    ux,uy=dx/length,dy/length
    d.polygon([q,(q[0]-17*ux+8*uy,q[1]-17*uy-8*ux),(q[0]-17*ux-8*uy,q[1]-17*uy+8*ux)],fill=color)

def note(d, xy, text):
    for i,line in enumerate(text.split('\n')):
        d.text((xy[0],xy[1]+34*i),line,font=FONTS[22],fill='#536277')

im,d=canvas('三种图像恢复路径：状态起点与条件输入', '概念示意；实线表示状态演化，虚线表示额外观测条件。',820)
rows=[('A  高斯噪声起点','随机高斯噪声\n独立于当前 LQ',BLUE),('B  直接观测起点','LQ 潜变量\n或预处理结果',GREEN),('C  残差增强路径','LQ 与高斯噪声\n加权混合状态',ORANGE)]
for i,(label,start,color) in enumerate(rows):
    y=190+i*195
    d.text((55,y+15),label,font=FONTS[28],fill=color)
    box(d,(365,y,690,y+105),start,color)
    box(d,(840,y,1175,y+105),'条件恢复网络\n一次或多次更新',color)
    box(d,(1320,y,1610,y+105),'恢复 HQ\n预测结果',color)
    arrow(d,(690,y+53),(840,y+53),color)
    arrow(d,(1175,y+53),(1320,y+53),color)
    note(d,(870,y+130),'LQ 特征／语义／其他条件')
    arrow(d,(1020,y+127),(1020,y+107),color,True)
note(d,(55,770),'B 中显式 Flow 与一步恢复映射不同；C 中 RRF 与残差 DDPM 不同。图中不表示真实空间距离。')
im.save(IMAGES/'fig_01_overview.png')

im,d=canvas('方案 A：高斯起点，LQ 通过条件约束恢复','训练可以访问 HQ；推理只访问 LQ，并另外采样初始噪声。',760)
d.text((55,165),'训练',font=FONTS[32],fill=BLUE)
box(d,(200,155,490,275),'HQ 目标\n＋随机高斯噪声')
box(d,(620,155,950,275),'构造中间状态\n目标与噪声插值')
box(d,(1080,155,1425,275),'预测速度／噪声\n计算监督与正则')
arrow(d,(490,215),(620,215));arrow(d,(950,215),(1080,215))
box(d,(680,325,1130,410),'LQ 编码与结构／语义特征',GREEN)
arrow(d,(1130,360),(1250,275),GREEN,True)
d.text((55,470),'推理',font=FONTS[32],fill=BLUE)
box(d,(200,460,490,580),'新采样高斯噪声\n初始生成状态')
box(d,(620,460,1030,580),'以 LQ 为条件\n反向恢复一次或多次')
box(d,(1190,460,1490,580),'恢复 HQ\n解码结果')
arrow(d,(490,520),(620,520));arrow(d,(1030,520),(1190,520))
arrow(d,(845,410),(845,460),GREEN,True)
note(d,(200,650),'VOSR、TEASR：Flow 速度参数化；SIED：标准加噪扩散。\n条件输入中存在 LQ，并不表示生成状态从 LQ 开始。')
im.save(IMAGES/'fig_02_gaussian_start.png')

im,d=canvas('方案 B：直接观测起点，两种训练形式','共同点是从 LQ 表示恢复；差别在是否显式学习连续 LQ–HQ Flow。',760)
d.text((55,170),'B1  显式 Flow',font=FONTS[28],fill=GREEN)
box(d,(355,155,660,275),'训练：LQ 与 HQ\n构造线性路径',GREEN)
box(d,(800,155,1135,275),'学习恢复速度\n与一致性等约束',GREEN)
box(d,(1290,155,1590,275),'推理：从 LQ\n积分到 HQ',GREEN)
arrow(d,(660,215),(800,215),GREEN);arrow(d,(1135,215),(1290,215),GREEN)
note(d,(355,315),'代表：FlowSR、RAR。直接路径也可以多步更新。')
d.text((55,450),'B2  一步映射',font=FONTS[28],fill=BLUE)
box(d,(355,435,660,555),'LQ 编码\n与可选跨帧融合')
box(d,(800,435,1135,555),'扩散骨干一次预测\n与端点换算')
box(d,(1290,435,1590,555),'HQ 潜变量\n与图像解码')
arrow(d,(660,495),(800,495));arrow(d,(1135,495),(1290,495))
note(d,(355,600),'代表：OSEDiff、HYPIR、DLoRAL。\n用数据／感知／分布损失训练，不等同于 B1 的整条速度监督。\n正则分支加噪，不代表生成器从随机噪声起步。')
im.save(IMAGES/'fig_03_direct_start.png')

im,d=canvas('方案 C：RRF 的残差路径与精确加速点','训练用真实残差构造状态；推理通过消去 HQ 系数获得可计算起点。',850)
d.text((55,160),'训练',font=FONTS[32],fill=ORANGE)
box(d,(230,145,555,265),'LQ 与 HQ 配对\n计算恢复残差',ORANGE)
box(d,(720,145,1080,265),'残差＋HQ＋噪声\n构造路径与速度目标',ORANGE)
box(d,(1230,145,1610,265),'有效区间内训练\n数据与分布约束',ORANGE)
arrow(d,(555,205),(720,205),ORANGE);arrow(d,(1080,205),(1230,205),ORANGE)
box(d,(540,330,1170,425),'代数推导：令未知 HQ 在状态中的系数为零',GREEN)
arrow(d,(860,265),(860,330),GREEN,True)
d.text((55,510),'推理',font=FONTS[32],fill=ORANGE)
box(d,(230,490,555,610),'已知 LQ\n＋新采样高斯噪声',ORANGE)
box(d,(720,490,1080,610),'加权混合起点\n默认半 LQ、半噪声',ORANGE)
box(d,(1230,490,1610,610),'从加速点反向积分\n得到 HQ',ORANGE)
arrow(d,(555,550),(720,550),ORANGE);arrow(d,(1080,550),(1230,550),ORANGE)
arrow(d,(860,425),(860,490),GREEN,True)
note(d,(230,675),'RRF：原始时间区间缩短，默认从半时刻开始。\n归一化时间后，它等价于 HQ 与特定带噪 LQ 端点之间的直线插值。\nResfusion 也加入残差，但采用离散 DDPM，噪声系数与预测目标不同。')
im.save(IMAGES/'fig_04_residual_start.png')

report_prefixes={
    'VOSR':'VOSR-', 'FlowSR':'Fast Image Super-Resolution',
    'HYPIR':'Harnessing Diffusion-Yielded', 'DLoRAL':'One-Step Diffusion for Detail-Rich',
    'ScaleResfusion':'ScaleResfusion-', 'Resfusion':'Resfusion-',
}
doc=HERE/(HERE.name+'.md')
content=doc.read_text(encoding='utf-8')
for key,prefix in report_prefixes.items():
    matches=[p for p in ROOT.rglob('*.md') if p.name.startswith(prefix) and not p.name.endswith('_sample.md')]
    if len(matches)!=1:
        raise RuntimeError(f'{key}: expected one report, found {matches}')
    target=quote(Path(os.path.relpath(matches[0],HERE)).as_posix(),safe='/')
    content=content.replace('@@REPORT:'+key+'@@',f'[独立解读报告]({target})')
doc.write_text(content,encoding='utf-8')
print('Generated 4 original PNG diagrams; resolved 6 existing report links.')
