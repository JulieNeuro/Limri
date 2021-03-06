fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
ver,
try,
        %% Generated by nipype.interfaces.spm
        if isempty(which('spm')),
             throw(MException('SPMCheck:NotFound', 'SPM not in matlab path'));
        end
        [name, version] = spm('ver');
        fprintf('SPM version: %s Release: %s\n',name, version);
        fprintf('SPM path: %s\n', which('spm'));
        spm('Defaults','fMRI');

        if strcmp(name, 'SPM8') || strcmp(name(1:5), 'SPM12'),
           spm_jobman('initcfg');
           spm_get_defaults('cmdline', 1);
        end

        jobs{1}.spm.spatial.normalise.write.subj.def = {...
'/volatile/7Li/data_test/08_09_21/transfo_rlink/y_combine.nii';...
};
jobs{1}.spm.spatial.normalise.write.subj.resample = {...
'/volatile/7Li/data_test/08_09_21/data_rlink/Li/01004RL20210430M03trufi_S005.nii,1';...
};
jobs{1}.spm.spatial.normalise.write.woptions.bb(1,1) = -42.0;
jobs{1}.spm.spatial.normalise.write.woptions.bb(1,2) = -68.0;
jobs{1}.spm.spatial.normalise.write.woptions.bb(1,3) = -35.0;
jobs{1}.spm.spatial.normalise.write.woptions.bb(2,1) = 78.0;
jobs{1}.spm.spatial.normalise.write.woptions.bb(2,2) = 76.0;
jobs{1}.spm.spatial.normalise.write.woptions.bb(2,3) = 85.0;
jobs{1}.spm.spatial.normalise.write.woptions.interp = 4;
jobs{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

        spm_jobman('run', jobs);

        
        if strcmp(name, 'SPM8') || strcmp(name(1:5), 'SPM12'),
            close('all', 'force');
        end;
            
,catch ME,
fprintf(2,'MATLAB code threw an exception:\n');
fprintf(2,'%s\n',ME.message);
if length(ME.stack) ~= 0, fprintf(2,'File:%s\nName:%s\nLine:%d\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;
end;