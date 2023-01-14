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

