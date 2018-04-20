clear
load('hmdb_idt_split1.mat');
[~, pred] = max(scores_idt');
acc = mean(pred' == gt_idt);

MK=importdata('/data/lilin/lilin_code/zjg_code/tle/data/hmdb51_splits/testlist01.txt');
M = size(MK,1);
tabo = zeros(M);
list = vn_idt;
idt_hmdb_test_score_1 = [];

for i=1:M
    tt = MK{i};
    ttt = tt(1:end-4);
    ii = 0;
    for j=1:M
        xx = list{j};
        if (strcmpi(xx, ttt) & (tabo(j) == 0))
            idt_hmdb_test_score_1 = [idt_hmdb_test_score_1 scores_idt(j,:)' ];
            j
            ii = 1;
            tabo(j) = 1;
            break
        end
    end
    if (ii==0)
        xxx
        S
    end
end
save('idt_hmdb_test_score_1.mat','idt_hmdb_test_score_1');