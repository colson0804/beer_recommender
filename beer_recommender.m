function beer_recommender()
    % Load data from our excel spreadsheet
    % M.data = numerical data
    % M.textdata = column and row headers
    M = importdata('beer_data.csv', ',', 1);
    d = M.data;
    h = M.textdata;
    
    % do some test plots
    scatter(d(:,3), d(:,4));
    
end