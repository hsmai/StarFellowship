"""HE 246개 태스크 -> 검출 프롬프트 자동 생성.

태스크명(스네이크케이스)에 조작 대상 객체가 그대로 들어 있어, 동사·전치사·로봇부위를
제거하면 객체 명사구가 남는다. description은 보조 확인용.
GroundingDINO는 프롬프트의 각 구를 독립 후보로 내므로 1~3개까지만 넣는다.
"""
import json, re

VERB = {
    "put","pick","place","open","close","push","pull","grab","hold","use","using","uses","wipe","clean",
    "fold","press","compress","click","adjust","tilt","center","extend","collapse","flip","remove","insert",
    "unplug","pour","drag","walk","walks","turn","turns","hand","pass","help","wear","stand","swing","slide",
    "lift","move","take","get","give","bring","carry","stack","unstack","swipe","wave","shake","tap","squeeze",
    "twist","rotate","drop","throw","catch","plug","hang","wipes","picks","holds","closes","opens","pushes",
    "pulls","grabs","places","puts","de-compress","decompress","erase","erases","sweep","scoop","pluck","tear",
    "zip","unzip","lock","unlock","switch","toggle","dial","type","write","draw","cut","peel","stir","mix",
    "spray","dust","brush","scrub","mop","vacuum","load","unload","transfer","deliver","handover","apply",
    "attach","detach","connect","disconnect","fill","empty","dump","toss","flick","poke","nudge","rest",
    "organize","align","pack","start","boiling","hit","sort","arrange","tidy","gather","collect","separate",
    "flatten","straighten","assemble","disassemble","reach","point","touch","knock","tug","hoist","lower",
    "raise","spin","roll","fetch","retrieve","return","hands","handing","picking","placing","holding",
}
STOP = {
    "a","an","the","into","onto","on","in","to","from","with","of","and","at","out","up","down","off",
    "towards","toward","away","back","its","it","his","her","their","robot","robots","hand","hands","left",
    "right","finger","fingers","thumb","arm","arms","g1","h1","then","after","before","by","for","over",
    "under","another","other","same","this","that","these","those","is","are","be","being","been","was",
    "were","while","when","where","which","who","whom","whose","what","how","why","also","again","twice",
    "once","times","time","upwards","downwards","sideways","around","near","next","closest","area","kept",
    "keeps","placed","placing","so","as","one","two","three","some","any","all","both","each","every","no",
    "not","very","just","only","own","inside","outside","against","agaist","loosely","upright","new","old",
    "small","large","big","long","short","empty","full","open","closed","first","second","third","last",
}
# 물체가 아닌 추상/부위 명사 — 단독으로 남으면 제거 (복합어의 일부면 유지)
NONOBJ = {
    "angle","lid","door","button","key","handle","cover","zipper","top","bottom","side","sides","middle",
    "front","edge","position","direction","way","step","action","motion","state","mode","level","height",
    "width","length","size","shape","color","order","surface","space","opening","base","piece","pieces","part",
}


def objects_from_taskname(task_name):
    base = task_name.split("/")[-1]
    base = re.sub(r"^h1-|^g1-", "", base)
    toks = [t for t in re.split(r"[_\-\s]+", base.lower()) if t]
    objs, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in VERB or t in STOP:
            i += 1
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if nxt and nxt not in VERB and nxt not in STOP:      # 복합어 결합 (phone stand, air pump)
            objs.append(f"{t} {nxt}")
            i += 2
        else:
            objs.append(t)
            i += 1
    out, seen = [], set()
    for o in objs:
        last = o.split()[-1]
        if " " not in o and (last in NONOBJ or len(o) <= 2):   # 단독 추상명사 제거
            continue
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out[:3] or ["object"]


def build(he_root="/data2/humanoid_dataset_isangmin/humanoid-everyday"):
    """반환: {task_index: {"task","category","prompt","targets"}}"""
    tasks = [json.loads(l) for l in open(f"{he_root}/meta/tasks.jsonl")]
    table = {}
    for t in tasks:
        objs = objects_from_taskname(t["task"])
        # 매칭키: 각 객체구의 마지막 단어(GroundingDINO 출력 phrase에 대개 포함됨)
        targets = {}
        for o in objs:
            k = o.split()[-1]
            if k not in targets:
                targets[k] = o
        table[t["task_index"]] = dict(task=t["task"], category=t["category"],
                                      prompt=" . ".join(objs) + " .", targets=targets)
    return table


if __name__ == "__main__":
    tb = build()
    print(f"태스크 {len(tb)}개")
    for ti in list(tb)[:20]:
        print(f"  [{tb[ti]['category']:12s}] {tb[ti]['task'].split('/')[-1][:42]:42s} -> {tb[ti]['prompt']}")
