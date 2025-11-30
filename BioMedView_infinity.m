classdef BioMedView_infinity < matlab.apps.AppBase
    % CLEAN, ERROR-FREE VERSION WITH 3D VIEW + ROI + EXPORT (BioMedView_infinity)

    properties (Access = public)
        UIFigure             matlab.ui.Figure
        GridLayout           matlab.ui.container.GridLayout

        % Layout Containers
        LeftPanel            matlab.ui.container.Panel
        RightPanel           matlab.ui.container.Panel
        TabGroup             matlab.ui.container.TabGroup

        % Axes (Visualization)
        MainAxes             matlab.ui.control.UIAxes
        HistAxes             matlab.ui.control.UIAxes

        % Data Storage
        RawVolume
        CurrentSliceImage
        ProcessedImage
        DicomInfo
        NumSlices double
        CurrentSliceIdx double = 1

        % Controls
        LoadButton           matlab.ui.control.Button
        SliceSlider          matlab.ui.control.Slider
        SliceLabel           matlab.ui.control.Label

        % Tab: Intensity
        ContrastSlider       matlab.ui.control.Slider
        BrightnessSlider     matlab.ui.control.Slider
        EnhanceDropDown      matlab.ui.control.DropDown

        % Tab: Filters & Edges
        NoiseFilterDropDown  matlab.ui.control.DropDown
        EdgeDetectDropDown   matlab.ui.control.DropDown

        % Tab: Segmentation
        ThresholdSlider      matlab.ui.control.Slider
        SegmentSwitch        matlab.ui.control.Switch
        RadonButton          matlab.ui.control.Button

        % ---- NEW FEATURES ----
        ROIHandle handle = []        % <- fixed: provide default empty value
        ROIStatsArea         matlab.ui.control.TextArea
        DrawROIRectButton    matlab.ui.control.Button
        DrawROIPolyButton    matlab.ui.control.Button
        DrawROIFreeButton    matlab.ui.control.Button
        ROIClearButton       matlab.ui.control.Button
        ExportImageButton    matlab.ui.control.Button
        ExportDicomButton    matlab.ui.control.Button
        Volume3DButton       matlab.ui.control.Button
    end

    methods (Access = private)

        %% ------------------ MAIN UPDATE FUNCTION ------------------
        function updateImage(app)
            if isempty(app.RawVolume)
                return;
            end

            % Extract slice
            if app.NumSlices > 1
                app.CurrentSliceImage = squeeze(app.RawVolume(:,:,app.CurrentSliceIdx));
            else
                app.CurrentSliceImage = squeeze(app.RawVolume);
            end
            img = double(app.CurrentSliceImage);

            % Normalize
            img = (img - min(img(:))) / (max(img(:)) - min(img(:)) + eps);

            % Filtering
            switch app.NoiseFilterDropDown.Value
                case 'Median (3x3)'
                    img = medfilt2(img,[3 3]);
                case 'NLM (Denoise)'
                    img = imnlmfilt(img);
            end

            % Enhancement
            switch app.EnhanceDropDown.Value
                case 'Hist Equalization'
                    img = histeq(img);
                case 'CLAHE'
                    img = adapthisteq(img);
            end

            % Windowing
            img = img * app.ContrastSlider.Value + app.BrightnessSlider.Value;
            img = min(max(img,0),1);

            % Edge detection
            if ~strcmp(app.EdgeDetectDropDown.Value,'None')
                img = double(edge(img,lower(app.EdgeDetectDropDown.Value)));
            end

            % Segmentation
            if strcmp(app.SegmentSwitch.Value,'On')
                img = double(img > app.ThresholdSlider.Value);
            end

            app.ProcessedImage = img;

            % DISPLAY
            imshow(app.ProcessedImage,[],'Parent',app.MainAxes);
            title(app.MainAxes,['Slice: ' num2str(app.CurrentSliceIdx) '/' num2str(app.NumSlices)]);

            % Histogram
            histogram(app.HistAxes,app.ProcessedImage(:),50);
            title(app.HistAxes,'Intensity Histogram');
        end

        %% -------------- ROI HANDLING -------------------
        function attachROI(app,roi)
            if ~isempty(app.ROIHandle) && isvalid(app.ROIHandle)
                delete(app.ROIHandle);
            end

            app.ROIHandle = roi;
            % Some ROI classes fire different events; use general listener on 'MovingROI' if available,
            % but to be robust, attach to addlistener for 'ROIMoved' and also listen for 'Moved' as fallback.
            try
                addlistener(roi,'ROIMoved',@(~,~)app.computeROIStats());
            catch
                try
                    addlistener(roi,'MovingROI',@(~,~)app.computeROIStats());
                catch
                    % ignore if event doesn't exist; user can press compute
                end
            end
            app.computeROIStats();
        end

        function computeROIStats(app)
            if isempty(app.ROIHandle) || ~isvalid(app.ROIHandle)
                app.ROIStatsArea.Value = {''};
                return;
            end
            % createMask works with images.roi.* objects when given axes handle
            try
                mask = createMask(app.ROIHandle, app.MainAxes);
            catch
                try
                    mask = app.ROIHandle.createMask();
                catch
                    app.ROIStatsArea.Value = {'ROI mask creation failed'};
                    return;
                end
            end

            vals = app.ProcessedImage(mask);
            if isempty(vals)
                app.ROIStatsArea.Value = {'No ROI'};
                return;
            end
            mu = mean(vals);
            sd = std(vals);
            med = median(vals);
            areaPix = sum(mask(:));

            app.ROIStatsArea.Value = {
                sprintf('Mean: %.4f',mu)
                sprintf('Std: %.4f',sd)
                sprintf('Median: %.4f',med)
                sprintf('Area (px): %d',areaPix)
            };
        end

        %% ---------------- EXPORT FUNCTIONS -----------------
        function ExportImage(app)
            if isempty(app.ProcessedImage)
                uialert(app.UIFigure,'No Processed Image','Export');
                return;
            end
            [f,p] = uiputfile({'*.png';'*.jpg';'*.tif'},'Save Image');
            if f==0; return; end
            out = uint8(255 * mat2gray(app.ProcessedImage));
            imwrite(out,fullfile(p,f));
        end

        function ExportDicom(app)
            if isempty(app.ProcessedImage)
                uialert(app.UIFigure,'No Processed Image','Export');
                return;
            end
            [f,p] = uiputfile('*.dcm','Save Processed Slice as DICOM');
            if f==0; return; end
            out = uint16(mat2gray(app.ProcessedImage) * 65535);
            if ~isempty(app.DicomInfo)
                dicomwrite(out,fullfile(p,f),app.DicomInfo,'CreateMode','Copy');
            else
                dicomwrite(out,fullfile(p,f));
            end
        end

        %% ---------------- 3D VIEW -------------------
        function show3D(app)
            if isempty(app.RawVolume)
                uialert(app.UIFigure,'Load a volume first.','3D View');
                return;
            end
            fig = figure('Name','3D Volume');
            try
                volshow(app.RawVolume);
            catch
                V = double(app.RawVolume);
                V = (V - min(V(:))) / (max(V(:)) - min(V(:)) + eps);
                try
                    Vs = smooth3(V);
                catch
                    Vs = V;
                end
                p = patch(isosurface(Vs,0.5));
                isonormals(Vs,p);
                p.FaceColor = 'red'; p.EdgeColor = 'none';
                daspect([1 1 1]); view(3); camlight; lighting gouraud;
            end
        end

        %% ---------------- RADON (helper) -------------------
        function RadonButtonPushed(app)
            if isempty(app.ProcessedImage)
                uialert(app.UIFigure,'No processed image','Radon');
                return;
            end
            I = app.ProcessedImage;
            theta = 0:180;
            [R,xp] = radon(I,theta);
            f = figure('Name','Radon Transform','NumberTitle','off');
            subplot(1,2,1); imshow(I,[]); title('Input');
            subplot(1,2,2); imagesc(theta,xp,R); colormap(gca,'turbo'); colorbar; xlabel('\theta (deg)'); ylabel('x''');
        end

        %% ---------------- CALLBACKS -------------------
        function Load(app,~)
            [file,path] = uigetfile({'*.dcm';'*.ima';'*.mat'});
            if file==0; return; end
            fp = fullfile(path,file);
            try
                if endsWith(file,'.mat')
                    S = load(fp);
                    fn = fieldnames(S);
                    app.RawVolume = squeeze(S.(fn{1}));
                    app.DicomInfo = [];
                else
                    app.DicomInfo = dicominfo(fp);
                    app.RawVolume = squeeze(dicomread(app.DicomInfo));
                end

                dims = size(app.RawVolume);
                if numel(dims)==3
                    app.NumSlices = dims(3);
                    app.SliceSlider.Enable = 'on';
                    app.SliceSlider.Limits = [1 app.NumSlices];
                    app.SliceSlider.Value = 1;
                else
                    app.NumSlices = 1;
                    app.SliceSlider.Enable = 'off';
                end
                app.CurrentSliceIdx = 1;
                app.updateImage();
            catch ME
                uialert(app.UIFigure,ME.message,'Loading Error');
            end
        end

        function SliceChanged(app,event)
            app.CurrentSliceIdx = round(event.Value);
            app.SliceLabel.Text = ['Slice: ' num2str(app.CurrentSliceIdx)];
            app.updateImage();
        end

        function Control(app,~)
            app.updateImage();
        end

    end

    %% --------------------- APP SETUP --------------------------
    methods (Access = public)
        function app = BioMedView_infinity

            app.UIFigure = uifigure('Visible','off');
            app.UIFigure.Position = [100 100 1100 650];
            app.UIFigure.Name = 'BioMedView_infinity — Enhanced';
            app.UIFigure.Color = [0.15 0.15 0.15];

            app.GridLayout = uigridlayout(app.UIFigure,[1 2]);
            app.GridLayout.ColumnWidth = {320,'1x'};

            %% LEFT PANEL
            app.LeftPanel = uipanel(app.GridLayout,'Title','Control Center');
            app.LeftPanel.BackgroundColor = [0.2 0.2 0.2];
            lp = uigridlayout(app.LeftPanel,[4 1]);

            %% LOAD BUTTON
            app.LoadButton = uibutton(lp,'push','Text','Load DICOM');
            app.LoadButton.ButtonPushedFcn = @(~,~)app.Load();

            %% SLICE
            nav = uipanel(lp,'Title','Slice Navigation'); ng = uigridlayout(nav,[2 1]);
            app.SliceSlider = uislider(ng,'ValueChangedFcn',@(s,e)app.SliceChanged(e));
            app.SliceSlider.Enable = 'off';
            app.SliceLabel = uilabel(ng,'Text','Slice: -');

            %% TABS
            app.TabGroup = uitabgroup(lp);

            % ENHANCE TAB
            t1 = uitab(app.TabGroup,'Title','Enhance');
            g1 = uigridlayout(t1,[6 1]);
            uilabel(g1,'Text','Contrast'); app.ContrastSlider = uislider(g1,'Limits',[0.1 3],'Value',1,'ValueChangedFcn',@(~,~)app.Control());
            uilabel(g1,'Text','Brightness'); app.BrightnessSlider = uislider(g1,'Limits',[-0.5 0.5],'Value',0,'ValueChangedFcn',@(~,~)app.Control());
            uilabel(g1,'Text','Enhancement'); app.EnhanceDropDown = uidropdown(g1,'Items',{'None','Hist Equalization','CLAHE'},'ValueChangedFcn',@(~,~)app.Control());

            % FILTER TAB
            t2 = uitab(app.TabGroup,'Title','Filters');
            g2 = uigridlayout(t2,[6 1]);
            uilabel(g2,'Text','Denoise'); app.NoiseFilterDropDown = uidropdown(g2,'Items',{'None','Median (3x3)','NLM (Denoise)'},'ValueChangedFcn',@(~,~)app.Control());
            uilabel(g2,'Text','Edges'); app.EdgeDetectDropDown = uidropdown(g2,'Items',{'None','Sobel','Prewitt','Canny'},'ValueChangedFcn',@(~,~)app.Control());

            % ANALYSIS TAB
            t3 = uitab(app.TabGroup,'Title','Analysis');
            g3 = uigridlayout(t3,[6 1]);
            uilabel(g3,'Text','Segmentation'); app.SegmentSwitch = uiswitch(g3,'Items',{'Off','On'},'ValueChangedFcn',@(~,~)app.Control());
            uilabel(g3,'Text','Threshold'); app.ThresholdSlider = uislider(g3,'Limits',[0 1],'Value',0.5,'ValueChangedFcn',@(~,~)app.Control());
            app.RadonButton = uibutton(g3,'push','Text','Radon','ButtonPushedFcn',@(s,e)app.RadonButtonPushed());

            % TOOLS TAB
            t4 = uitab(app.TabGroup,'Title','Tools');
            g4 = uigridlayout(t4,[8 1]);

            app.DrawROIRectButton = uibutton(g4,'push','Text','Draw Rectangle ROI','ButtonPushedFcn',@(~,~)app.attachROI(drawrectangle(app.MainAxes)));
            app.DrawROIPolyButton = uibutton(g4,'push','Text','Draw Polygon ROI','ButtonPushedFcn',@(~,~)app.attachROI(drawpolygon(app.MainAxes)));
            app.DrawROIFreeButton = uibutton(g4,'push','Text','Draw Freehand ROI','ButtonPushedFcn',@(~,~)app.attachROI(drawfreehand(app.MainAxes)));
            app.ROIClearButton = uibutton(g4,'push','Text','Clear ROI','ButtonPushedFcn',@(s,e)deleteSafeROI(app));

            app.Volume3DButton = uibutton(g4,'push','Text','3D Volume View','ButtonPushedFcn',@(~,~)app.show3D());
            app.ExportImageButton = uibutton(g4,'push','Text','Export PNG/JPG','ButtonPushedFcn',@(~,~)app.ExportImage());
            app.ExportDicomButton = uibutton(g4,'push','Text','Export DICOM','ButtonPushedFcn',@(~,~)app.ExportDicom());

            app.ROIStatsArea = uitextarea(g4,'Editable','off','Value',{'ROI Stats'});

            %% RIGHT PANEL
            app.RightPanel = uipanel(app.GridLayout,'BackgroundColor',[0 0 0]);
            rp = uigridlayout(app.RightPanel,[2 1]);
            rp.RowHeight = {'3x','1x'};

            app.MainAxes = uiaxes(rp);
            app.MainAxes.Color = [0 0 0];
            app.MainAxes.XColor = 'none';
            app.MainAxes.YColor = 'none';

            app.HistAxes = uiaxes(rp);
            app.HistAxes.Color = [0.1 0.1 0.1];

            app.UIFigure.Visible = 'on';

            % Local helper to safely delete ROI
            function deleteSafeROI(appLocal)
                try
                    if ~isempty(appLocal.ROIHandle) && isvalid(appLocal.ROIHandle)
                        delete(appLocal.ROIHandle);
                        appLocal.ROIHandle = [];
                    end
                catch
                end
            end
        end
    end
end
