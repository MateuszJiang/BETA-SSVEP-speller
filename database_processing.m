
electrodes = 64;
blocks = 4;
trials = 40;

directories = [];
str = 'dir_trial';
for i = 1 : trials
    directories{i} = sprintf('%s_%d',str,i);
    %mkdir testData directories{i}
    mkdir(sprintf('divided_database/dir_trial_%d',i))
end

results = [];
for k = 1:blocks
    for l = 1:numfiles
        for m =1:trials
            for n = 1:electrodes
                results(n,:) = DataSets{l}.data.EEG(n,:,k,m);
            end
            for n = 1:electrodes
                if size(results) == [n,750]
                    for x = 751:1000
                        results(n,x) = 0;
                    end
                end
            end
            fileName = sprintf('subject_%d_block_%d.mat', l ,k);
            path  = sprintf('divided_database/%s/%s', directories{m}, fileName);
            save( path , 'results')
            results = [];
        end
    end
end