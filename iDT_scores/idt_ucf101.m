clear
load('ucf_idt_split3.mat');
[~, pred] = max(scores_idt');
acc = mean(pred' == gt_idt);

MK=importdata('/data/lilin/lilin_code/zjg_code/tle/data/ucf101_splits/testlist03.txt');
M = size(MK,1);
tabo = zeros(M);
list = vn_idt;
idt_ucf_test_score_3 = [];

for i=1:M
    S = regexp(MK{i}, '/', 'split');
    tt = S{2};
    ttt = tt(1:end-4);
    ii = 0;
    for j=1:M
        xx = list{j};
        if (strcmpi(xx, ttt) & (tabo(j) == 0))
            idt_ucf_test_score_3 = [idt_ucf_test_score_3 scores_idt(j,:)' ];
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
save('idt_ucf_test_score_3.mat','idt_ucf_test_score_3');