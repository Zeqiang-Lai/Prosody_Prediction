import re
import thulac

def tokenize(raw_path):
    thu1 = thulac.thulac()
    with open(raw_path, 'r') as f:
        data = f.read()
        new_data = re.sub('#\d', '', data)
        lines = new_data.split('\n')
        
        with open('tokenized.txt', 'w') as f:
            for i in range(0, len(lines), 2):
                line = lines[i].split('\t')
                id = line[0]
                sent = line[1]
                text = thu1.cut(sent, text=True)
                f.write(text + '\n')

                if i % 200 == 0:
                    print("Process {0}.".format(i))

def extract_text(raw_path):
    with open(raw_path, 'r') as f:
        lines = f.readlines()
        lines = lines[::2]
        with open('text.txt', 'w') as fout:
            for line in lines:
                line = line.split('\t')
                fout.write(line[1])

def split_text(text_path):
    
    def split_line(lines, separator, file):
        parts = []
        for line in lines:
            parts += line.strip().split(separator)
        parts_ = [re.sub('#\d', '', part) for part in parts]
        file.write(' '.join(parts_) + '\n')
        return parts

    with open(text_path, 'r') as f:
        lines = f.readlines()

        f1 = open('text1.txt', 'w')
        f2 = open('text2.txt', 'w')
        f3 = open('text3.txt', 'w')
        f4 = open('text4.txt', 'w')
        for line in lines:
            line = split_line([line], '#4', f4)
            line = split_line(line, '#3', f3)
            line = split_line(line, '#2', f2)
            split_line(line, '#1', f1)

        f1.close()
        f2.close()
        f3.close()
        f4.close()

def tag(words_path, tokenzied_path, tag):
    f = open(tokenzied_path, 'r')
    lines = f.readlines()
    f2 = open(words_path, 'r')
    lines2 = f2.readlines()

    fout = open('final_tag_'+tag+'.txt', 'w')

    def compute_range(items):
        cur = 0
        ranges = []
        start = []
        end = []
        for item in items:
            ranges.append((cur, cur+len(item)))
            start.append(cur)
            end.append(cur+len(item))
            cur += len(item)
        return start, end

    for tok, sep in zip(lines2, lines):
        toks = tok.strip().split(' ')
        seps = sep.strip().split(' ')
        tok_s, tok_e = compute_range(toks)
        sep_s, sep_e = compute_range(seps)

        pairs = []
        for i in range(len(toks)):
            if(tok_s[i] in sep_s):
                pairs.append(toks[i] + '_B')
            elif(tok_e[i] in sep_e):
                pairs.append(toks[i] + '_I')
            else:
                pairs.append(toks[i] + '_I')
        fout.write(' '.join(pairs) + '\n')
    

def split_pos(tokenzied_path):
    with open(tokenzied_path, 'r') as f:
        lines = f.readlines()
        f = open('words.txt', 'w')
        for line in lines:
            parts = line.strip().split(' ')
            parts = [part.split('_')[0] for part in parts]
            f.write(' '.join(parts) + '\n')

if __name__ == "__main__":
    raw_path = 'data/biaobei/processed/000001-010000.txt'
    # tokenize()
    # extract_text(raw_path)
    # split_text('data/biaobei/text.txt')
    # split_pos('data/biaobei/processed/tokenized.txt')
    words_path = 'data/biaobei/processed/words.txt'
    tokenzied_path = 'data/biaobei/processed/text3.txt'
    tag(words_path, tokenzied_path, '3')