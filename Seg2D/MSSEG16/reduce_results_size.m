% There is no space left on my computer, estimate ranges based on reachable
% set and save those instead

resFiles = dir("results/*.mat");

for i = 1:height(resFiles)
    fName = "results/" + resFiles(i).name;
    try
        data = load(fName);
        try
            [lb,ub] = data.R.estimateRanges;
            rT = data.rT;
            save(fName, "ub", "lb", "rT");
        catch ME
            warning(fName);
            warning(ME.message);
        end
    catch ME
        warning(fName);
        warning(ME.message);
    end
end