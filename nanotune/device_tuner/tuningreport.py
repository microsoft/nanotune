import smtplib
import os
import logging
import textwrap
from datetime import datetime

from typing import Optional, List, Dict, Union, Any

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders

from reportlab.lib.pagesizes import A4

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch  # , cm
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle

import nanotune as nt
from nanotune.data.dataset import Dataset
from nanotune.utils import flatten_list, get_param_values

logger = logging.getLogger(__name__)

title_style = ParagraphStyle(
    "title_style", alignment=0, fontSize=12, fontName="Times-Roman"
)

body_style = ParagraphStyle(
    "body_style",
    alignment=0,
    fontSize=10,
    fontName="Times-Roman",
    justifyBreaks=0,
    splitLongWords=1,
    rightIndent=10,
)
rpstyles = getSampleStyleSheet()

aW = 460
aH = 800

left_margin = 50
right_margin = 30
top_margin = 30
bottom_margin = 40
h_padding = 40
v_padding = 40


class TuningReport():
    """"""

    def __init__(
        self,
        stage_summaries: Optional[Dict[str, List[List[int]]]],
        receiver: List[str],
        # = ['me@onenote.com', 'nanotune@outlook.com'],
        sender: Optional[str],
        #   = 'nanotune@outlook.com',
        sender_pwd: Optional[str],  # = 'N@notune',
        db_name: Optional[str] = None,
        db_folder: Optional[str] = None,
        guid: Optional[str] = None,
        comments: Optional[Dict[str, str]] = None,
        device_name: str = "nt device",
        section: Optional[str] = None,
        subject: Optional[str] = None,
        smalltalk: str = "",
        server: str = "smtp-mail.outlook.com",
        port: int = 587,
        pdf_folder: Optional[str] = None,
        pdf_name: str = "nanotune_results",
    ):
        """"""
        if db_name is None:
            db_name, _ = nt.get_database()
        if db_folder is None:
            db_folder = nt.config["db_folder"]

        if "me@onenote.com" not in receiver:
            receiver.append("me@onenote.com")

        self.receiver = receiver
        self.sender = sender
        self.smalltalk = smalltalk
        self.stage_summaries = stage_summaries
        self.comments = {} if comments is None else comments
        self.db_name = db_name
        self.device_name = device_name
        self.db_folder = db_folder

        if section is None:
            self.section = " @" + self.device_name
        else:
            self.section = "@Measurements"

        self.subject = subject
        if self.subject is None:
            self.subject = ""
        self.server = server
        self.port = port
        self.sender_pwd = sender_pwd
        self.pdf_name = pdf_name
        if pdf_folder is not None:
            self.pdf_folder = pdf_folder
        else:
            self.pdf_folder = os.path.join(
                self.db_folder, "tuning_results", self.device_name, "pdfs"
            )
            if not os.path.exists(self.pdf_folder):
                os.makedirs(self.pdf_folder)

        self.fields = (
            ("name", "Parameter"),
            ("value", "Value"),
        )

        self.column_ratio = [0.55, 0.35]
        # [figure_width, table_width] * total

    def press(self):
        """"""
        self.summarize()
        self.create_pdf()
        # self.distribute()

    def summarize(self):
        """
        generates list of files to send

        fig saved with
        self.dir = os.path.join(nt.config['db_folder'], 'tuning_results',
                                    self.device_name)

        filename = str(self.ds.guid)
        """
        self.smalltalk += "\n Data IDs in this bundle: \n"
        self._files = {}
        inv_dict = {}
        # sort IDs to make sure pdfs are printed in same oder as they were
        # taken
        for k, v in self.stage_summaries.items():
            for qc_id in flatten_list(v):
                inv_dict[qc_id] = k
        sorted_ids = list(flatten_list(self.stage_summaries.values()))
        sorted_ids.sort(key=int)
        # for stage, value in self.stage_summaries.items():
        for qc_run_id in sorted_ids:
            # stage = inv_dict[qc_run_id]
            # if stage[0:7] == 'failed_':
            #     stage = stage[7:]
            #     try:
            #         s = self.comments[qc_run_id]
            #     except KeyError:
            #         s = ''
            #     self.comments[qc_run_id] = 'Classified as poor result.\n' + s
            ds = Dataset(qc_run_id, self.db_name)
            device_name = ds.device_name
            f_folder = os.path.join(self.db_folder, "tuning_results", device_name)
            # for qc_run_id in flatten_list(value):
            self.smalltalk += str(qc_run_id) + ", "

            # filename = stage + '_fit_ds'
            # filename += str(qc_run_id) + '.png'
            filename = os.path.join(f_folder, str(ds.ds.guid) + ".png")

            self._files[str(qc_run_id)] = filename

    def create_pdf(self):
        """
        fig saved with
        self.dir = os.path.join(nt.config['db_folder'], 'tuning_results',
                                    self.device_name)

        filename = str(self.ds.guid)
        """

        my_datetime = datetime.now()
        self.pdf_name = (
            self.pdf_name + "_" + my_datetime.strftime("%H%M_%d%m%Y") + ".pdf"
        )
        fig_width = aW * self.column_ratio[0]

        clm_width_meta = (aW * self.column_ratio[1]) / len(self.fields)

        c = canvas.Canvas(os.path.join(self.pdf_folder, self.pdf_name), pagesize=A4)

        for qc_run_id, fig_file in sorted(self._files.items()):
            (param_values, feature_values) = get_param_values(
                qc_run_id, self.db_name, return_meta_add_on=True
            )

            comment = self.subject + "<br/>"
            # c.saveState()
            title = "Dataset " + qc_run_id

            # Prepare header
            header = Paragraph(title, title_style)
            h_w, h_h = header.wrap(aW, aH)

            # Prepare image
            img = ImageReader(fig_file)
            im_width, im_height = img.getSize()
            aspect = im_height / float(im_width)
            fig_height = fig_width * aspect

            # Prepare metadata section

            meta_table = Table(
                param_values,
                colWidths=[clm_width_meta] * len(self.fields),
                hAlign="CENTER",
                rowHeights=0.22 * inch,
            )
            meta_table.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONT", (0, 1), (-1, -1), "Helvetica"),
                        ("LINEBELOW", (0, 0), (1, 0), 0.08, colors.black),
                        ("SIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                        # ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        ("ALIGN", (1, 1), (1, -1), "LEFT"),
                        ("INNERGRID", (0, 0), (-1, -1), 0.08, colors.beige),
                        # ('BOX', (0,0), (-1,-1), 0.25, colors.grey),
                    ]
                )
            )

            meta_width, meta_height = meta_table.wrap(aW - im_width, aH / 2)

            # Prepare comments header
            comments_header = Paragraph("Comments:", title_style)
            avail_height = aH - fig_height - v_padding
            comm_h_width, comm_h_height = comments_header.wrap(
                im_width, avail_height  # aW - meta_width,
            )
            # Prepare comments
            my_datetime = datetime.now()
            ts = "Printed on " + my_datetime.strftime("%c")

            try:
                data_specific_comment = self.comments[int(qc_run_id)]
                comment += data_specific_comment + "<br/>"
                comment += self.comments["general"] + "<br/>"

                comment += self.smalltalk + "<br/>"
            except Exception as e:
                logger.warning(
                    "Unable to summarize result of " + "dataset {}".format(qc_run_id)
                )
                pass
            comment_ts = comment + ts
            comment_ts = textwrap.fill(comment_ts, 70)
            comment_ts = comment_ts.replace("\n", "<br/>")

            comments_p = Paragraph(comment_ts, body_style)

            avail_height = aH - fig_height - v_padding - comm_h_height

            comm_width, comm_height = comments_p.wrap(im_width, avail_height)  # aW,

            line_widths = comments_p.getActualLineWidths0()
            number_of_lines = len(line_widths)
            if number_of_lines > 1:
                actual_width = comm_height
            if number_of_lines == 1:
                actual_width = min(line_widths)
                comm_width, comm_height = comments_p.wrap(im_width, avail_height)

            # Prepare features
            feat_table = Table(
                feature_values,
                colWidths=[clm_width_meta] * len(self.fields),
                hAlign="CENTER",
                rowHeights=0.22 * inch,
            )
            feat_table.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONT", (0, 1), (-1, -1), "Helvetica"),
                        ("LINEBELOW", (0, 0), (1, 0), 0.08, colors.black),
                        ("SIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                        # ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        ("ALIGN", (1, 1), (1, -1), "LEFT"),
                        ("INNERGRID", (0, 0), (-1, -1), 0.08, colors.beige),
                        # ('BOX', (0,0), (-1,-1), 0.25, colors.grey),
                    ]
                )
            )
            avail_height = aH - meta_height  # fig_height - v_padding - comm_h_height
            avail_height -= comm_height
            feat_width, feat_height = feat_table.wrap(aW - im_width, avail_height)

            # Draw everyting on canvas

            header.drawOn(c, left_margin, aH - top_margin)

            c.drawImage(
                img,
                left_margin,
                aH - top_margin - fig_height - v_padding,
                width=fig_width * 1.1,
                height=fig_height * 1.1,
                mask="auto",
            )

            meta_table.drawOn(
                c,
                left_margin + fig_width + h_padding,
                aH - meta_height - top_margin / 2,  #  - v_padding
            )

            comments_header.drawOn(
                c,
                left_margin,
                aH
                - top_margin
                - comm_h_height
                - fig_height
                - 2 * v_padding,  # - add_on_height
            )

            comments_p.drawOn(
                c,
                left_margin,
                aH
                - top_margin
                - comm_h_height
                - comm_height
                - fig_height
                - 2 * v_padding
                - comm_h_height,  # - add_on_height
            )

            feat_table.drawOn(
                c,
                left_margin + fig_width + h_padding,
                aH - meta_height - top_margin / 2 - feat_height - v_padding,
                # top_margin - fig_height - 2*v_padding - feat_height
            )

            # new page
            c.showPage()
            c.saveState()

        c.save()

    def distribute(self):
        """"""
        self.message = MIMEMultipart()
        self.message["From"] = self.sender
        self.message["To"] = ", ".join(self.receiver)
        self.message["Date"] = formatdate(localtime=True)
        self.message["Subject"] = self.subject + self.section
        self.message.attach(MIMEText(self.smalltalk))

        part = MIMEBase("application", "octet-stream")
        pdf_file = os.path.join(self.pdf_folder, self.pdf_name)
        with open(pdf_file, "rb") as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            'attachment; filename="{}"'.format(os.path.basename(self.pdf_name)),
        )
        self.message.attach(part)

        smtp = smtplib.SMTP(self.server, self.port)
        smtp.starttls()
        smtp.login(self.sender, self.sender_pwd)
        smtp.sendmail(self.sender, self.receiver, self.message.as_string())
        smtp.quit()
