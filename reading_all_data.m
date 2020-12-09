filenames =["S1.mat", "S2.mat", "S3.mat", "S4.mat", "S5.mat", "S6.mat", "S7.mat", "S8.mat", "S9.mat", "S10.mat",
            "S11.mat", "S12.mat", "S13.mat", "S14.mat", "S15.mat", "S16.mat", "S17.mat", "S18.mat", "S19.mat", "S20.mat" 
            "S21.mat", "S22.mat", "S23.mat", "S24.mat", "S25.mat", "S26.mat", "S27.mat", "S28.mat", "S29.mat", "S30.mat"
            "S31.mat", "S32.mat", "S33.mat", "S34.mat", "S35.mat", "S36.mat", "S37.mat", "S38.mat", "S39.mat", "S40.mat"
            "S41.mat", "S42.mat", "S43.mat", "S44.mat", "S45.mat", "S46.mat", "S47.mat", "S48.mat", "S49.mat", "S50.mat"
            "S51.mat", "S52.mat", "S53.mat", "S54.mat", "S55.mat", "S56.mat", "S57.mat", "S58.mat", "S59.mat", "S60.mat"
            "S61.mat", "S62.mat", "S63.mat", "S64.mat", "S65.mat", "S66.mat", "S67.mat", "S68.mat", "S69.mat", "S70.mat"];

numfiles = 70;
DataSets = cell(1, numfiles);

for k = 1:numfiles
  DataSets{k} = load(filenames{k});
end