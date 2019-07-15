%11692 subjects for training and 896 subjects for prediction. Since I use regression method, I can only use the last visit feature to predict. And I used age and month as features and set Nan features as 0.

%read TADPOLE_Table from the form TADPOLE_D1_D2.csv. 
read;
%%
%covert the feature to matrix. 
RID=table2array(TADPOLE_Table(:,1));
YEAR=cellfun(@str2double,table2array(TADPOLE_Table(:,93)));
AGE=cellfun(@str2double,table2array(TADPOLE_Table(:,12)))+YEAR;
UCSFFSL=cellfun(@str2double,table2array(TADPOLE_Table(:,124:468)));
UCSFFSX=cellfun(@str2double,table2array(TADPOLE_Table(:,488:832)));
FDG=cellfun(@str2double,table2array(TADPOLE_Table(:,19)));
AV45=cellfun(@str2double,table2array(TADPOLE_Table(:,21)));
AV45all=cellfun(@str2double,table2array(TADPOLE_Table(:,1175:1412)));
AV1451=cellfun(@str2double,table2array(TADPOLE_Table(:,21)));
AV1451all=cellfun(@str2double,table2array(TADPOLE_Table(:,1415:1656)));
APOE=cellfun(@str2double,table2array(TADPOLE_Table(:,18)));
ADA11=cellfun(@str2double,table2array(TADPOLE_Table(:,23)));
ADA13=table2array(TADPOLE_Table(:,24));%
MMSE=cellfun(@str2double,table2array(TADPOLE_Table(:,25)));
RAVLT=cellfun(@str2double,table2array(TADPOLE_Table(:,26:29)));
MOCA=cellfun(@str2double,table2array(TADPOLE_Table(:,31)));
Ecog=cellfun(@str2double,table2array(TADPOLE_Table(:,32:45)));
Ventricle=table2array(TADPOLE_Table(:,48));%
Hippocampus=cellfun(@str2double,table2array(TADPOLE_Table(:,49)));
ICV=cellfun(@str2double,table2array(TADPOLE_Table(:,54)));
Month=cellfun(@str2double,table2array(TADPOLE_Table(:,95)));
Ventricle_ICV=Ventricle./ICV;%
Hippocampus_ICV=Hippocampus./ICV;
DX=table2array(TADPOLE_Table(:,55));%
DX_num=zeros(12741,1);
for i=1:12741
    if(strcmp(DX(i),'NL')||strcmp(DX(i),'MCI to NL'))
        DX_num(i)=1;
    elseif(strcmp(DX(i),'MCI')||strcmp(DX(i),'NL to MCI')||strcmp(DX(i),'Dementia to MCI'))
        DX_num(i)=2;
    elseif(strcmp(DX(i),'Dementia')||strcmp(DX(i),'NL to Dementia')||strcmp(DX(i),'MCI to Dementia'))
        DX_num(i)=3;
    else
        DX_num(i)=0;
    end
end

ABeta=cellfun(@str2double,table2array(TADPOLE_Table(:,1903)));
TAU=cellfun(@str2double,table2array(TADPOLE_Table(:,1904)));
PTAU=cellfun(@str2double,table2array(TADPOLE_Table(:,1905)));
%%
ADA13(isnan(ADA13))=0;
Ventricle_ICV(isnan(Ventricle_ICV))=0;
APOE(isnan(APOE))=-1;

% AB=zeros(12741,1);
% for i=1:12741
%     if(~isempty(ABeta{i}))
%         AB(i)=str2double(ABeta(i));
%     end
% end
% T=zeros(12741,1);
% for i=1:12741
%     if(~isempty(TAU{i}))
%         T(i)=str2double(TAU(i));
%     end
% end
% PT=zeros(12741,1);
% for i=1:12741
%     if(~isempty(PTAU{i}))
%         PT(i)=str2double(PTAU(i));
%     end
% end
maxy=max(YEAR);
NYEAR=YEAR/maxy;
maxa=max(AGE);
NAGE=AGE/maxa;
Ventricle_ICV(Ventricle_ICV>1)=0;
%testdata=[ADA13,DX_num,Ventricle_ICV,ADA11,AV45,AV45all,AV1451,AV1451all,FDG,UCSFFSL,UCSFFSX,MMSE,RAVLT,MOCA,Ecog,Hippocampus_ICV,ABeta,TAU,PTAU,APOE];
%select feature to train 3 models. 
testdata=[ADA13,Ventricle_ICV,DX_num,ADA11,AV45,AV1451,FDG,MMSE,RAVLT,MOCA,Ecog,Hippocampus_ICV,ABeta,TAU,PTAU,APOE];

data=sum(~isnan(testdata));
col=prod(~isnan(testdata),2);
testdata_zero=testdata;
testdata_zero(isnan(testdata_zero))=0;
testdata_zero=zscore(testdata_zero);
testdata_zero=[NYEAR,NAGE,testdata_zero];

testdata_ADA13=[ADA13,testdata_zero(:,1:2),testdata_zero(:,4:end)];
testdata_VentICV=[Ventricle_ICV,testdata_zero(:,1:3),testdata_zero(:,5:end)];
testdata_DX=[DX_num,testdata_zero(:,1:4),testdata_zero(:,6:end)];


%% Delete the features of which the label is null.
j=1;
for i=1:12741
   if(ADA13(i)==0)
       testdata_ADA13(j,:)=[];
   else
       j=j+1;
   end
end
j=1;
for i=1:12741
    if(Ventricle_ICV(i)==0)
       testdata_VentICV(j,:)=[];
   else
       j=j+1;
   end
end
j=1;
for i=1:12741
    if(DX_num(i)==0)
       testdata_DX(j,:)=[];
   else
       j=j+1;
   end
end
%testdata_nozero


%% build structure for predict subjects
load D2;%predict subjects
idyear=[RID,YEAR,ADA13,Ventricle_ICV];
s(896)=struct('id',0,'y',-1,'line',0,'month',zeros(60),'adas13',zeros(60),'ladas13',0,'venticv',zeros(60),'lventice',0,'dx',zeros(60),'feat',[]);
for i=1:896
    s(i).id=D2(i);
    s(i).y=-1;
    s(i).line=0;
    s(i).ladas13=0;
    s(i).lventice=0;
    for j=1:60
        s(i).month(j)=j/12/maxy;
    end
end
for i=1:12741
    for j=1:896
        if idyear(i,1)==s(j).id
            if idyear(i,2)>s(j).y
                s(j).y=idyear(i,2);
                s(j).line=i;
            end
            if s(j).ladas13<idyear(i,3)
               s(j).ladas13=idyear(i,3);
            end     
            if s(j).lventice<idyear(i,4)
                s(j).lventice=idyear(i,4);
            end  
        end
    end
end

for i=1:896
    s(i).feat=testdata_zero(s(i).line,:);
end

%% train 3 models
trainmodel_adas13=trainRegressionModel(testdata_ADA13);
trainmodel_venticv=trainRegressionModel(testdata_VentICV);
trainmodel_dx=trainRegressionModel(testdata_DX);
%calculate confident intervals
inter_adas13 = coefCI(trainmodel_adas13.LinearModel,0.5);
inter_venticv = coefCI(trainmodel_venticv.LinearModel,0.5);
inter_dx = coefCI(trainmodel_dx.LinearModel,0.5);
%% predict the subjects
for i=1:896
    for j=1:60
        s(i).adas13(j)=trainmodel_adas13.predictFcn([s(i).feat(1)+s(i).month(j),s(i).feat(2)+s(i).month(j),s(i).feat(4:end)]);
        s(i).venticv(j)=trainmodel_venticv.predictFcn([s(i).feat(1)+s(i).month(j),s(i).feat(2)+s(i).month(j),[s(i).feat(3),s(i).feat(5:end)]]);
        s(i).dx(j)=trainmodel_dx.predictFcn([s(i).feat(1)+s(i).month(j),s(i).feat(2)+s(i).month(j),[s(i).feat(3:4),s(i).feat(6:end)]]);
    end
end
for i=1:896
     s(i).dx= fliplr(s(i).dx);
end
    
%% Now construct the forecast spreadsheet and output it.

startDate = datenum('01-Jan-2018');
N_D2=896;

%* Create arrays to contain the 60 monthly forecasts for each D2 subject
nForecasts = 5*12; % forecast 5 years (60 months).
% 1. Clinical status forecasts
%    i.e. relative likelihood of NL, MCI, and Dementia (3 numbers)
CLIN_STAT_forecast = zeros(N_D2, nForecasts, 3);
% 2. ADAS13 forecasts 
%    (best guess, upper and lower bounds on 50% confidence interval)
ADAS13_forecast = zeros(N_D2, nForecasts, 3);
% 3. Ventricles volume forecasts 
%    (best guess, upper and lower bounds on 50% confidence interval)
Ventricles_ICV_forecast = zeros(N_D2, nForecasts, 3);
for i=1:896
    div_adas13=s(i).adas13(1)-s(i).ladas13;
    div_venticv=s(i).venticv(1)-s(i).lventice;
    for j=1:60
        ADAS13_forecast(i,j,1)=s(i).adas13(j)-div_adas13;
        ADAS13_forecast(i,j,2)=ADAS13_forecast(i,j,1)-inter_adas13(1,1)/10;
        ADAS13_forecast(i,j,3)=ADAS13_forecast(i,j,1)+inter_adas13(1,2)/10;
        Ventricles_ICV_forecast(i,j,1)=s(i).venticv(j)-div_venticv;
        Ventricles_ICV_forecast(i,j,2)=Ventricles_ICV_forecast(i,j,1)+inter_venticv(1,1);
        Ventricles_ICV_forecast(i,j,3)=Ventricles_ICV_forecast(i,j,1)-inter_venticv(1,2);
        ProNL=abs(s(i).dx(j)-3);
        ProMCI=abs(s(i).dx(j)-2);
        ProAD=abs(s(i).dx(j)-1);
        Prosum=ProNL+ProMCI+ProAD;
        CLIN_STAT_forecast(i,j,1)=ProNL/Prosum;
        CLIN_STAT_forecast(i,j,2)=ProMCI/Prosum;
        CLIN_STAT_forecast(i,j,3)=ProAD/Prosum;
    end
end


submission_table =  cell2table(cell(N_D2*nForecasts,12), ...
  'VariableNames', {'RID', 'ForecastMonth', 'ForecastDate', ...
  'CNRelativeProbability', 'MCIRelativeProbability', 'ADRelativeProbability', ...
  'ADAS13', 'ADAS1350_CILower', 'ADAS1350_CIUpper', ...
  'Ventricles_ICV', 'Ventricles_ICV50_CILower', 'Ventricles_ICV50_CIUpper' });
%* Repeated matrices - compare with submission template
submission_table.RID = reshape(repmat(D2, [1, nForecasts])', N_D2*nForecasts, 1);
submission_table.ForecastMonth = repmat((1:nForecasts)', [N_D2, 1]);
%* First subject's submission dates
for m=1:nForecasts
  submission_table.ForecastDate{m} = datestr(addtodate(startDate, m-1, 'month'), 'yyyy-mm');
end
%* Repeated matrices for submission dates - compare with submission template
submission_table.ForecastDate = repmat(submission_table.ForecastDate(1:nForecasts), [N_D2, 1]);

%* Pre-fill forecast data, encoding missing data as NaN
nanColumn = nan(size(submission_table.CNRelativeProbability));
submission_table.CNRelativeProbability = nanColumn;
submission_table.MCIRelativeProbability = nanColumn;
submission_table.ADRelativeProbability = nanColumn;
submission_table.ADAS13 = nanColumn;
submission_table.ADAS1350_CILower = nanColumn;
submission_table.ADAS1350_CIUpper = nanColumn;
submission_table.Ventricles_ICV = nanColumn;
submission_table.Ventricles_ICV50_CILower = nanColumn;
submission_table.Ventricles_ICV50_CIUpper = nanColumn;

%*** Paste in month-by-month forecasts **
%* 1. Clinical status
  %*  a) CN probabilities
col = 4;
t = CLIN_STAT_forecast(:,:,1)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
  %*  b) MCI probabilities
col = 5;
t = CLIN_STAT_forecast(:,:,2)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
  %*  c) AD probabilities
col = 6;
t = CLIN_STAT_forecast(:,:,3)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
%* 2. ADAS13 score
col = 7;
t = ADAS13_forecast(:,:,1)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
  %*  a) Lower and upper bounds (50% confidence intervals)
col = 8;
t = ADAS13_forecast(:,:,2)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
col = 9;
t = ADAS13_forecast(:,:,3)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
%* 3. Ventricles volume (normalised by intracranial volume)
col = 10;
t = Ventricles_ICV_forecast(:,:,1)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
  %*  a) Lower and upper bounds (50% confidence intervals)
col = 11;
t = Ventricles_ICV_forecast(:,:,2)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);
col = 12;
t = Ventricles_ICV_forecast(:,:,3)';
col = submission_table.Properties.VariableNames(col);
submission_table{:,col} = t(:);

%* Convert all numbers to strings
hdr = submission_table.Properties.VariableNames;
for k=1:length(hdr)
  if ~iscell(submission_table.(hdr{k}))
    %submission_table{1:10,hdr{k}} = varfun(@num2str,submission_table{1:10,hdr{k}},'OutPutFormat','cell');
    submission_table.(hdr{k}) = strrep(cellstr(num2str(submission_table{:,hdr{k}})),' ','');
  end
end

%* Use column names that match the submission template
columnNames = {'RID', 'Forecast Month', 'Forecast Date',...
'CN relative probability', 'MCI relative probability', 'AD relative probability',	...
'ADAS13',	'ADAS13 50% CI lower', 'ADAS13 50% CI upper', 'Ventricles_ICV', ...
'Ventricles_ICV 50% CI lower',	'Ventricles_ICV 50% CI upper'};
%* Convert table to cell array to write to file, line by line
%  This is necessary because of spaces in the column names: writetable()
%  doesn't handle this.
tableCell = table2cell(submission_table);
tableCell = [columnNames;tableCell];
%* Write file line-by-line
fid = fopen(outputFile,'w');
for i=1:size(tableCell,1)
  fprintf(fid,'%s\n', strjoin(tableCell(i,:), ','));
end
fclose(fid);


