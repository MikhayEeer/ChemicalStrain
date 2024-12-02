package main

import (
	"log"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"github.com/xuri/excelize/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	a := app.New()
	w := a.NewWindow("Excel to Line Chart")

	fileDialog := dialog.FileOpen(func(reader fyne.URIReadCloser, err error) {
		if err != nil {
			dialog.ShowError(err, w)
			return
		}
		defer reader.Close()

		excelFile, err := excelize.OpenReader(reader)
		if err != nil {
			dialog.ShowError(err, w)
			return
		}

		sheetName := excelFile.GetSheetName(0)
		rows, err := excelFile.GetRows(sheetName)
		if err != nil {
			dialog.ShowError(err, w)
			return
		}

		var xData, yData []float64
		for i, row := range rows {
			if i == 0 {
				continue // Skip header row
			}
			x, _ := excelize.EvalFormula(&excelize.Formula{Expression: row[0]})
			y, _ := excelize.EvalFormula(&excelize.Formula{Expression: row[1]})
			xData = append(xData, x.Value.Float())
			yData = append(yData, y.Value.Float())
		}

		p, err := plot.New()
		if err != nil {
			log.Fatal(err)
		}
		p.Title.Text = "Line Chart from Excel Data"
		p.X.Label.Text = "X Axis"
		p.Y.Label.Text = "Y Axis"

		line, err := plotter.NewLine(plotCoordinates(xData, yData))
		if err != nil {
			log.Fatal(err)
		}
		line.LineStyle.Width = vg.Points(1)
		p.Add(line)

		img, err := p.WriterToImage(vg.Inch*5, vg.Inch*5)
		if err != nil {
			log.Fatal(err)
		}

		imageCanvas := container.NewMax(img)
		w.SetContent(imageCanvas)
	}, w)

	openButton := fyne.NewMenuItem("Open Excel File", func() {
		fileDialog.Show()
	})

	menu := fyne.NewMenu("File", openButton)
	w.SetMainMenu(fyne.NewMainMenuBar(menu...))

	w.Resize(fyne.NewSize(800, 600))
	w.ShowAndRun()
}

func plotCoordinates(xData, yData []float64) plotter.XYs {
	pts := make(plotter.XYs, len(xData))
	for i := range pts {
		pts[i].X = xData[i]
		pts[i].Y = yData[i]
	}
	return pts
}
