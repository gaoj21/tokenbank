"""Fetch the Wikipedia articles linked by every FRAMES question (wiki_links), for the open-book setting.

Fifty titles per request (revisions, current wikitext), redirects followed, one request every two seconds.
The wikitext is rendered to text with mwparserfromhell, keeping what plain-text extracts drop: infoboxes
become "field: value" lines, tables one row per line with cells separated by " | ", and the common date,
unit and list templates are expanded; references, citations, files, categories and the reference sections
are dropped. Saves data/frames_wiki/<sha1(title)>.json with the requested title, the resolved title and the
text, and data/frames_wiki/.complete once every title has a file.
usage: python analysis/frames_fetch.py"""
import calendar, hashlib, json, os, re, time, urllib.parse, urllib.request

import mwparserfromhell as mw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'frames_wiki')
UA = {'User-Agent': 'climb-research/1.0 (academic research; batch of 50 titles per request)'}
DROP_SECTIONS = {'references', 'external links', 'notes', 'further reading', 'citations', 'sources', 'bibliography',
                 'footnotes', 'notes and references', 'explanatory notes'}
DATES = {'birth date', 'death date', 'birth date and age', 'death date and age', 'start date', 'end date',
         'start date and age', 'end date and age', 'film date', 'dob', 'dts', 'date', 'release date', 'death-date',
         'birth-date', 'birth date and age2', 'death date and age2', 'start and end dates', 'opening date'}
YEARS = {'birth year and age', 'death year and age', 'birth year', 'death year'}
LISTS = {'ubl', 'unbulleted list', 'plainlist', 'flatlist', 'hlist', 'plain list', 'flat list', 'bulleted list',
         'collapsible list', 'ordered list', 'enum', 'nowrap', 'nobr', 'small', 'big', 'resize', 'center'}
SKIP_FIELDS = {'image', 'image_size', 'imagesize', 'caption', 'alt', 'signature', 'signature_alt', 'logo',
               'logo_size', 'image_upright', 'module', 'embed', 'map', 'map_image', 'pushpin_map', 'coordinates',
               'image_caption', 'image_alt', 'website', 'url', 'signature_size', 'footnotes', 'onlysourced'}


def name_of(t):
    return re.sub(r'\s+', ' ', str(t.name).strip().replace('_', ' ')).lower()


def positional(t):
    return [p.value for p in t.params if not p.showkey]


def date_str(vals):
    nums = []
    for v in vals:
        s = str(v).strip()
        if re.fullmatch(r'\d{1,4}', s):
            nums.append(int(s))
        if len(nums) == 3:
            break
    if len(nums) == 3 and 1 <= nums[1] <= 12:
        return f'{calendar.month_name[nums[1]]} {nums[2]}, {nums[0]}'
    if len(nums) >= 2 and 1 <= nums[1] <= 12:
        return f'{calendar.month_name[nums[1]]} {nums[0]}'
    return str(nums[0]) if nums else ''


def template(t):
    n = name_of(t)
    if n.startswith(('infobox', 'taxobox', 'automatic taxobox', 'speciesbox', 'chembox', 'drugbox')) or n.endswith(' infobox'):
        lines = []
        for p in t.params:
            k = str(p.name).strip()
            if not p.showkey or k.lower() in SKIP_FIELDS:
                continue
            v = render(p.value).strip()
            if v:
                lines.append(f"{k.replace('_', ' ')}: {re.sub(r'\s*\n\s*', '; ', v)}")
        return '\n' + '\n'.join(lines) + '\n\n'
    if n in DATES:
        return date_str(positional(t))
    if n in YEARS:
        return date_str(positional(t)[:1])
    if n in ('convert', 'cvt'):
        v = [render(x).strip() for x in positional(t)]
        return ' '.join(v[:2]) if v else ''
    if n in LISTS:
        return ', '.join(x for x in (render(v).strip().replace('\n', ', ').lstrip('*, ') for v in positional(t)) if x)
    if n in ('lang', 'transl', 'nihongo', 'script'):
        v = positional(t)
        return render(v[1] if n in ('lang', 'transl', 'script') and len(v) > 1 else v[0]) if v else ''
    if n.startswith('lang-'):
        v = positional(t)
        return render(v[0]) if v else ''
    if n in ('abbr', 'abbrlink', 'sic', 'proper name', 'em', 'strong', 'mvar', 'math', 'var', 'smallcaps', 'sc',
             'not a typo', 'linktext', 'w', 'ill', 'interlanguage link', 'visible anchor'):
        v = positional(t)
        return render(v[0]) if v else ''
    if n in ('marriage', 'married'):
        v = [render(x).strip() for x in positional(t)]
        return (v[0] + (f" (m. {'–'.join(v[1:3])})" if len(v) > 1 else '')) if v else ''
    if n == 'sortname':
        v = [render(x).strip() for x in positional(t)]
        return ' '.join(v[:2])
    if n in ('flag', 'flagcountry', 'flagu', 'flag country', 'country'):
        v = positional(t)
        return render(v[0]) if v else ''
    if n in ('yes', 'no', 'partial', 'n/a', 'na', 'tba', 'tbd', 'won', 'nom', 'pending'):
        v = positional(t)
        return render(v[0]) if v else {'n/a': 'N/A', 'na': 'N/A', 'nom': 'Nominated'}.get(n, n.capitalize())
    if n in ('ndash', 'snd', 'spaced ndash'):
        return ' – '
    if n in ('mdash', 'spaced mdash'):
        return ' — '
    if n in ('frac', 'fraction', 'sfrac'):
        v = [render(x).strip() for x in positional(t)]
        return ' '.join(v[:-2] + ['/'.join(v[-2:])]) if len(v) >= 2 else ''.join(v)
    if n in ('us$', 'usd', 'gbp', '£', '€', 'inr', 'aud', 'cad'):
        v = [render(x).strip() for x in positional(t)]
        sym = {'us$': 'US$', 'usd': 'US$', 'gbp': '£', '£': '£', '€': '€', 'inr': '₹', 'aud': 'A$', 'cad': 'C$'}[n]
        return sym + (v[0] if v else '')
    return ''


def cell(c):
    return re.sub(r'\s+', ' ', render(c.contents)).strip()


def table(tag):
    rows, cur = [], []
    for c in tag.contents.nodes:
        if not isinstance(c, mw.nodes.Tag):
            continue
        tg = str(c.tag).lower()
        if tg == 'tr':
            if cur:
                rows.append(cur); cur = []
            r = [cell(x) for x in c.contents.nodes if isinstance(x, mw.nodes.Tag) and str(x.tag).lower() in ('td', 'th')]
            if any(r):
                rows.append(r)
        elif tg == 'td' and not cur and not rows and cell(c).startswith('+'):   # "|+" caption, parsed as a cell
            rows.append([cell(c)[1:].strip()])
        elif tg in ('td', 'th'):
            cur.append(cell(c))
        elif tg == 'caption':
            rows.append([cell(c)])
    if cur:
        rows.append(cur)
    return '\n' + '\n'.join(' | '.join(r) for r in rows if any(r)) + '\n\n'


def render(code):
    out = []
    for node in (code.nodes if hasattr(code, 'nodes') else []):
        if isinstance(node, mw.nodes.Text):
            out.append(str(node))
        elif isinstance(node, mw.nodes.Wikilink):
            title = str(node.title).strip()
            if re.match(r'(?i)\s*:?(file|image|category|media):', title):
                continue
            out.append(render(node.text) if node.text is not None else title.split('#')[0] or title)
        elif isinstance(node, mw.nodes.ExternalLink):
            out.append(render(node.title) if node.title is not None else '')
        elif isinstance(node, mw.nodes.Template):
            out.append(template(node))
        elif isinstance(node, mw.nodes.Tag):
            tg = str(node.tag).lower()
            if tg in ('ref', 'references', 'gallery', 'timeline', 'imagemap', 'templatestyles', 'score', 'graph',
                      'mapframe', 'maplink', 'syntaxhighlight', 'categorytree'):
                continue
            if tg == 'table':
                out.append(table(node))
            elif tg == 'br':
                out.append('\n')
            elif node.contents is not None:
                out.append(render(node.contents) + ('\n' if tg in ('li', 'p', 'div', 'dd', 'dt') else ''))
        elif isinstance(node, mw.nodes.Heading):
            out.append('\n' + render(node.title).strip() + '\n')
        elif isinstance(node, mw.nodes.HTMLEntity):
            out.append(node.normalize())
    return ''.join(out)


def plain(wikitext):
    code = mw.parse(wikitext)
    for sec in code.get_sections(levels=[2]):
        h = sec.filter_headings(recursive=False)
        if h and render(h[0].title).strip().lower() in DROP_SECTIONS:
            try:
                code.remove(sec)
            except ValueError:
                pass
    text = render(code)
    text = re.sub(r"\[\[|\]\]|\{\{|\}\}|''+", '', text)
    text = re.sub(r'[ \t ]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\(\s*[,;]?\s*\)', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def link_titles(links):
    """Article titles of one question's wiki_links, in order. A few entries hold several URLs joined by commas,
    and a few are percent-encoded twice."""
    out = []
    for u in links:
        for part in re.split(r',\s*(?=https?://)', u):
            if 'wikipedia.org/wiki/' not in part:
                continue
            t = part.split('/wiki/', 1)[1].split('#')[0].strip().rstrip(',').strip()
            t = urllib.parse.unquote(urllib.parse.unquote(t)).replace('_', ' ').strip()
            if t and t not in out:
                out.append(t)
    return out


def titles():
    from datasets import load_dataset
    ds = load_dataset('google/frames-benchmark', split='test')
    return sorted({t for it in ds for t in link_titles(eval(it['wiki_links']))})


path = lambda t: os.path.join(OUT, hashlib.sha1(t.encode()).hexdigest() + '.json')


def fetch_batch(batch):
    q = urllib.parse.urlencode(dict(action='query', prop='revisions', rvprop='content', rvslots='main', redirects=1,
                                    titles='|'.join(batch), format='json', formatversion=2))
    d = json.load(urllib.request.urlopen(urllib.request.Request('https://en.wikipedia.org/w/api.php?' + q, headers=UA), timeout=120))
    norm = {x['from']: x['to'] for x in d['query'].get('normalized', [])}
    red = {x['from']: x['to'] for x in d['query'].get('redirects', [])}
    pages = {p['title']: p for p in d['query']['pages']}
    out = {}
    for t in batch:
        n = norm.get(t, t); r = red.get(n, n); p = pages.get(r, {})
        rev = (p.get('revisions') or [{}])[0]
        out[t] = (r, (rev.get('slots') or {}).get('main', {}).get('content', ''))
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    T = titles(); todo = [t for t in T if not os.path.exists(path(t))]
    print(len(T), 'titles,', len(todo), 'to fetch', flush=True)
    empty = 0
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        for a in range(10):
            try:
                got = fetch_batch(batch); break
            except Exception as e:
                print('retry', k, type(e).__name__, str(e)[:60], flush=True); time.sleep(min(600, 15 * 2 ** a))
        else:
            print('batch failed', k, flush=True); continue
        for t, (r, w) in got.items():
            txt = plain(w) if w else ''
            json.dump(dict(title=t, resolved=r, text=txt), open(path(t), 'w')); empty += not txt
        print(k + len(batch), 'fetched', flush=True)
        time.sleep(2)
    if all(os.path.exists(path(t)) for t in T):
        open(os.path.join(OUT, '.complete'), 'w').write(f'{len(T)} titles, {empty} empty\n')
    print('done, empty', empty)
