% --------------vary k  experiment for fair-equality---------------
% time and embeddings are stored.

addpath(genpath('./baseline'));
ks = [2,3,4,5,6,7,8,9,10,20];
for i = 1:length(ks)
    k = ks(i);
    % facebookNet
    dataset_name = 'facebookNet';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    t1 = 0; tic;
    H = Fair_SC_normalized(adj,k,color);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H;
end

for i = 1:length(ks)
    k = ks(i);
    % german
    dataset_name = 'german';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    t1 = 0; tic;
    H = Fair_SC_normalized(adj,k,color);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H;
end

for i = 1:length(ks)
    k = ks(i);
    % SBM_1000
    dataset_name = 'SBM_1000';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    t1 = 0; tic;
    H = Fair_SC_normalized(adj,k,color);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H;
end

for i = 1:length(ks)
    k = ks(i);
    % dblp
    dataset_name = 'dblp';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    t1 = 0; tic;
    H = Fair_SC_normalized(adj,k,color);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H;
end

for k = 5 : 5 : 50
    % lastfm
    dataset_name = 'lastfm';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    t1 = 0; tic;
    H = Fair_SC_normalized(adj,k,color);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H;
end


% --------------vary k  experiment for fair-equality-s---------------
% time and embeddings are stored.

ks = [2,3,4,5,6,7,8,9,10,20];
for i = 1:length(ks)
    k = ks(i);
    % facebookNet
    dataset_name = 'facebookNet';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    n = size(adj, 1);
    sensitive = color;
    % converting sensitive to a vector with entries in [h] and building F %%%
    sens_unique=unique(sensitive);
    h = length(sens_unique);
    sens_unique=reshape(sens_unique,[1,h]);
    sensitiveNEW=sensitive;
    temp=1;
    for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
    end
    F=zeros(n,h-1);
    for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1;
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
    end
    degrees = sum(adj, 1);
    D = diag(degrees);
    %%%%
    
    t1 = 0; tic;
    H = alg3(adj,D,F,k);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for i = 1:length(ks)
    k = ks(i);
    % german
    dataset_name = 'german';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    n = size(adj, 1);
    sensitive = color;
    % converting sensitive to a vector with entries in [h] and building F %%%
    sens_unique=unique(sensitive);
    h = length(sens_unique);
    sens_unique=reshape(sens_unique,[1,h]);
    sensitiveNEW=sensitive;
    temp=1;
    for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
    end
    F=zeros(n,h-1);
    for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1;
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
    end
    degrees = sum(adj, 1);
    D = diag(degrees);
    %%%%
    
    t1 = 0; tic;
    H = alg3(adj,D,F,k);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for i = 1:length(ks)
    k = ks(i);
    % SBM_1000
    dataset_name = 'SBM_1000';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    n = size(adj, 1);
    sensitive = color;
    % converting sensitive to a vector with entries in [h] and building F %%%
    sens_unique=unique(sensitive);
    h = length(sens_unique);
    sens_unique=reshape(sens_unique,[1,h]);
    sensitiveNEW=sensitive;
    temp=1;
    for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
    end
    F=zeros(n,h-1);
    for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1;
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
    end
    degrees = sum(adj, 1);
    D = diag(degrees);
    %%%%
    
    t1 = 0; tic;
    H = alg3(adj,D,F,k);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for i = 1:length(ks)
    k = ks(i);
    % dblp
    dataset_name = 'dblp';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    n = size(adj, 1);
    sensitive = color;
    % converting sensitive to a vector with entries in [h] and building F %%%
    sens_unique=unique(sensitive);
    h = length(sens_unique);
    sens_unique=reshape(sens_unique,[1,h]);
    sensitiveNEW=sensitive;
    temp=1;
    for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
    end
    F=zeros(n,h-1);
    for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1;
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
    end
    degrees = sum(adj, 1);
    D = diag(degrees);
    %%%%
    
    t1 = 0; tic;
    H = alg3(adj,D,F,k);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for k = 5 : 5 : 50
    % lastfm
    dataset_name = 'lastfm';
    color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    color = color(:,2) + 1;
    m = max(color);
    edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
    EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable);
    adj = G.adjacency;

    n = size(adj, 1);
    sensitive = color;
    % converting sensitive to a vector with entries in [h] and building F %%%
    sens_unique=unique(sensitive);
    h = length(sens_unique);
    sens_unique=reshape(sens_unique,[1,h]);
    sensitiveNEW=sensitive;
    temp=1;
    for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
    end
    F=zeros(n,h-1);
    for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1;
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
    end
    degrees = sum(adj, 1);
    D = diag(degrees);
    %%%%
    
    t1 = 0; tic;
    H = alg3(adj,D,F,k);
    t1 = t1 + toc;
    writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
    writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
    clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for k = 5 : 5 : 50
     % credit_education
     dataset_name = 'credit_education';
     color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     color = color(:,2) + 1;
     m = max(color);
     edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
     G = graph(EdgeTable);
     adj = G.adjacency;

     n = size(adj, 1);
     sensitive = color;
     % converting sensitive to a vector with entries in [h] and building F %%%
     sens_unique=unique(sensitive);
     h = length(sens_unique);
     sens_unique=reshape(sens_unique,[1,h]);
     sensitiveNEW=sensitive;
     temp=1;
     for ell=sens_unique
         sensitiveNEW(sensitive==ell)=temp;
         temp=temp+1;
     end
     F=zeros(n,h-1);
     for ell=1:(h-1)
         temp=(sensitiveNEW == ell);
         F(temp,ell)=1;
         groupSize = sum(temp);
         F(:,ell) = F(:,ell)-groupSize/n;
     end
     degrees = sum(adj, 1);
     D = diag(degrees);
     %%%%

     t1 = 0; tic;
     H = alg3(adj,D,F,k);
     t1 = t1 + toc;
     writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
     writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
     clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for k = 5 : 5 : 50
     % deezer
     dataset_name = 'deezer';
     color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     color = color(:,2) + 1;
     m = max(color);
     edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
     G = graph(EdgeTable);
     adj = G.adjacency;

     n = size(adj, 1);
     sensitive = color;
     % converting sensitive to a vector with entries in [h] and building F %%%
     sens_unique=unique(sensitive);
     h = length(sens_unique);
     sens_unique=reshape(sens_unique,[1,h]);
     sensitiveNEW=sensitive;
     temp=1;
     for ell=sens_unique
         sensitiveNEW(sensitive==ell)=temp;
         temp=temp+1;
     end
     F=zeros(n,h-1);
     for ell=1:(h-1)
         temp=(sensitiveNEW == ell);
         F(temp,ell)=1;
         groupSize = sum(temp);
         F(:,ell) = F(:,ell)-groupSize/n;
     end
     degrees = sum(adj, 1);
     D = diag(degrees);
     %%%%

     t1 = 0; tic;
     H = alg3(adj,D,F,k);
     t1 = t1 + toc;
     writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
     writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
     clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end


for k = 5 : 5 : 50
     % pokec_age
     dataset_name = 'pokec_age';
     color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     color = color(:,2) + 1;
     m = max(color);
     edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
     G = graph(EdgeTable);
     adj = G.adjacency;

     n = size(adj, 1);
     sensitive = color;
     % converting sensitive to a vector with entries in [h] and building F %%%
     sens_unique=unique(sensitive);
     h = length(sens_unique);
     sens_unique=reshape(sens_unique,[1,h]);
     sensitiveNEW=sensitive;
     temp=1;
     for ell=sens_unique
         sensitiveNEW(sensitive==ell)=temp;
         temp=temp+1;
     end
     F=zeros(n,h-1);
     for ell=1:(h-1)
         temp=(sensitiveNEW == ell);
         F(temp,ell)=1;
         groupSize = sum(temp);
         F(:,ell) = F(:,ell)-groupSize/n;
     end
     degrees = sum(adj, 1);
     D = diag(degrees);
     %%%%

     t1 = 0; tic;
     H = alg3(adj,D,F,k);
     t1 = t1 + toc;
     writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
     writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
     clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

for k = 5 : 5 : 50
     % pokec_sex
     dataset_name = 'pokec_sex';
     color = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_colors.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     color = color(:,2) + 1;
     m = max(color);
     edges = readmatrix(['datasets/processed/',dataset_name,'/',dataset_name,'_edges.csv'], NumHeaderLines=0,Delimiter=',',OutputType='int32');
     EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
     G = graph(EdgeTable);
     adj = G.adjacency;

     n = size(adj, 1);
     sensitive = color;
     % converting sensitive to a vector with entries in [h] and building F %%%
     sens_unique=unique(sensitive);
     h = length(sens_unique);
     sens_unique=reshape(sens_unique,[1,h]);
     sensitiveNEW=sensitive;
     temp=1;
     for ell=sens_unique
         sensitiveNEW(sensitive==ell)=temp;
         temp=temp+1;
     end
     F=zeros(n,h-1);
     for ell=1:(h-1)
         temp=(sensitiveNEW == ell);
         F(temp,ell)=1;
         groupSize = sum(temp);
         F(:,ell) = F(:,ell)-groupSize/n;
     end
     degrees = sum(adj, 1);
     D = diag(degrees);
     %%%%

     t1 = 0; tic;
     H = alg3(adj,D,F,k);
     t1 = t1 + toc;
     writematrix(H,['./results/embeddings/', 'fair_equality_s_','embeddings_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv']);
     writematrix(t1,['./results/embeddings/', 'fair_equality_s_','t1_',dataset_name,'_k',int2str(k),'_m',int2str(m),'.csv'])
     clear color edges EdgeTable G adj H sensitive sens_unique sensitiveNEW F D degrees;
end

